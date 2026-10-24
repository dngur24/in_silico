"""
Security helpers for external bioinformatics web-server crawlers.

Protects the backend against compromised / defaced upstream PHP servers
(e.g., PHP2Shell RCE, malicious meta-refresh or HTTP 30x redirects to
internal/private networks, unbounded response bombs, and rogue third-party
requests or drive-by downloads in headless Chromium).
"""

from __future__ import annotations

import ipaddress
import logging
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB ceiling for HTML responses
DEFAULT_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def build_allowed_hosts(base_url: str, extra_hosts: Optional[Iterable[str]] = None) -> set[str]:
    """
    Build the set of trusted hostnames derived from ``base_url``
    (including its ``www.`` / bare-domain counterpart if applicable).
    """
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"Cannot extract hostname from base_url: {base_url!r}")

    hosts = {host}
    # Allow bare <-> www.* transitions on the same domain, but never for IP literals
    try:
        ipaddress.ip_address(host)
    except ValueError:
        bare = host[4:] if host.startswith("www.") else host
        if "." in bare:
            hosts.add(bare)
            hosts.add(f"www.{bare}")

    if extra_hosts:
        for h in extra_hosts:
            if h:
                hosts.add(h.lower())
    return hosts


def validate_safe_url(url: str, allowed_hosts: set[str]) -> str:
    """
    Ensure ``url`` uses http/https, carries no embedded credentials, does not
    target private/loopback/link-local IPs, and matches ``allowed_hosts``.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Blocked unsafe URL scheme '{parsed.scheme}' in {url!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"Blocked URL containing credentials: {url!r}")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"Blocked URL without hostname: {url!r}")

    # Explicitly reject loopback / private / link-local / multicast IP literals
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"Blocked non-public IP address in URL: {url!r}")
    except ValueError as exc:
        if "Blocked non-public IP" in str(exc):
            raise

    if host not in allowed_hosts:
        raise ValueError(
            f"Blocked untrusted redirect/target host '{host}' "
            f"(allowed: {sorted(allowed_hosts)})"
        )
    return url


def resolve_safe_redirect(
    redirect_target: str,
    current_url: str,
    allowed_hosts: set[str],
) -> str:
    """
    Resolve a relative or absolute redirect target (from HTTP Location header
    or HTML meta-refresh) against ``current_url`` and validate its host.
    """
    target = redirect_target.strip()
    base = current_url if current_url.endswith("/") or "." in current_url.rsplit("/", 1)[-1] else f"{current_url}/"
    resolved = urljoin(base, target)
    return validate_safe_url(resolved, allowed_hosts)


def safe_request_text(
    session: requests.Session,
    method: str,
    url: str,
    *,
    allowed_hosts: set[str],
    timeout: float = 30.0,
    max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    **kwargs,
) -> str:
    """
    Execute an HTTP request while:
      1. Validating the initial URL and every HTTP 30x redirect against ``allowed_hosts``
      2. Enforcing a strict ``max_bytes`` ceiling on the response body to prevent memory DoS
    """
    current_url = validate_safe_url(url, allowed_hosts)
    current_method = method.upper()
    req_kwargs = dict(kwargs)
    req_kwargs.pop("allow_redirects", None)
    req_kwargs.pop("stream", None)

    for hop in range(max_redirects + 1):
        resp = session.request(
            current_method,
            current_url,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
            **req_kwargs,
        )
        try:
            if resp.status_code in _REDIRECT_STATUSES and "Location" in resp.headers:
                if hop >= max_redirects:
                    raise ValueError(f"Exceeded max HTTP redirects ({max_redirects}) for {url!r}")
                next_url = resolve_safe_redirect(
                    resp.headers["Location"],
                    current_url,
                    allowed_hosts,
                )
                logger.debug("Following validated HTTP %d redirect → %s", resp.status_code, next_url)
                current_url = next_url
                if resp.status_code in (301, 302, 303) and current_method != "GET":
                    current_method = "GET"
                    req_kwargs.pop("data", None)
                    req_kwargs.pop("files", None)
                    req_kwargs.pop("json", None)
                continue

            resp.raise_for_status()

            content_len = resp.headers.get("Content-Length")
            if content_len and content_len.isdigit() and int(content_len) > max_bytes:
                raise ValueError(
                    f"Upstream response Content-Length ({content_len} bytes) "
                    f"exceeds safety limit ({max_bytes} bytes)"
                )

            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise ValueError(
                        f"Upstream response body exceeded safety limit ({max_bytes} bytes)"
                    )

            encoding = resp.encoding or "utf-8"
            try:
                return buf.decode(encoding, errors="replace")
            except LookupError:
                return buf.decode("utf-8", errors="replace")
        finally:
            resp.close()

    raise ValueError(f"Unexpected redirect loop for {url!r}")


def create_hardened_playwright_page(
    browser,
    *,
    allowed_hosts: set[str],
    user_agent: str,
):
    """
    Create a hardened Playwright browser context and page that:
      - Disables file downloads (blocks drive-by downloads)
      - Blocks Service Workers
      - Aborts all network requests to non-whitelisted hosts or non-HTTP(S) schemes
        (except harmless inline ``data:`` images/fonts)
    Returns ``(context, page)``.
    """
    context = browser.new_context(
        user_agent=user_agent,
        accept_downloads=False,
        service_workers="block",
    )

    def _route_guard(route):
        req = route.request
        req_url = req.url
        parsed = urlparse(req_url)

        if parsed.scheme == "data" and req.resource_type in ("image", "font"):
            route.continue_()
            return

        try:
            validate_safe_url(req_url, allowed_hosts)
        except ValueError as exc:
            logger.debug("Playwright blocked outbound request %s: %s", req_url, exc)
            route.abort()
            return

        route.continue_()

    context.route("**/*", _route_guard)
    page = context.new_page()
    return context, page
