"""
DBAASP v3 REST API client & bioactivity lookup predictor.

Target API: https://dbaasp.org/peptides
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.security import build_allowed_hosts, safe_request_text

logger = logging.getLogger(__name__)

_BASE_URL = "https://dbaasp.org"
_SEARCH_URL = f"{_BASE_URL}/peptides"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


@dataclass
class PredictionResult:
    peptide: str
    method: str = "Exact"
    score: Optional[float] = None
    label: Optional[str] = None
    properties: Optional[dict[str, Any]] = None
    records: list[dict[str, Any]] = field(default_factory=list, repr=False)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class DBAASPClient:
    """Queries DBAASP v3 peptide database by amino-acid sequence."""

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        max_retries: int = 3,
        retry_backoff: float = 0.8,
        inter_request_delay: float = 0.3,
    ) -> None:
        self.timeout = timeout
        self.inter_request_delay = inter_request_delay
        self._allowed_hosts = build_allowed_hosts(_BASE_URL)
        self._session = self._build_session(max_retries, retry_backoff)

    def search(self, sequence: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return matching DBAASP peptide records for ``sequence``."""
        seq = self._clean(sequence)
        err = self._validate(seq)
        if err:
            raise ValueError(err)

        qs = urlencode({"sequence.value": seq, "limit": limit, "offset": 0})
        url = f"{_SEARCH_URL}?{qs}"
        raw = safe_request_text(
            self._session,
            "GET",
            url,
            allowed_hosts=self._allowed_hosts,
            timeout=self.timeout,
        )
        payload = json.loads(raw)
        if isinstance(payload, dict):
            if "errors" in payload and payload["errors"]:
                raise RuntimeError("; ".join(str(e) for e in payload["errors"]))
            data = payload.get("data", [])
            return data if isinstance(data, list) else []
        return payload if isinstance(payload, list) else []

    def predict(self, sequence: str, *, method: str = "Exact") -> PredictionResult:
        """
        Check whether ``sequence`` is recorded as an antimicrobial/bioactive peptide in DBAASP v3.

        method:
          - 'Exact'     : exact full-sequence match in DBAASP
          - 'Substring' : any DBAASP entry containing ``sequence`` as a motif/subsequence
        """
        seq = self._clean(sequence)
        mode = "Substring" if str(method).lower().startswith("sub") else "Exact"
        err = self._validate(seq)
        if err:
            return PredictionResult(peptide=seq, method=mode, error=err)

        try:
            qs = urlencode({"sequence.value": seq, "limit": 25, "offset": 0})
            raw = safe_request_text(
                self._session,
                "GET",
                f"{_SEARCH_URL}?{qs}",
                allowed_hosts=self._allowed_hosts,
                timeout=self.timeout,
            )
            payload = json.loads(raw)
            items = payload.get("data", []) if isinstance(payload, dict) else []
            total_sub = int(payload.get("totalCount", len(items))) if isinstance(payload, dict) else len(items)

            exact_matches = [
                it for it in items
                if str(it.get("sequence") or "").strip().upper() == seq
            ]
            matched = items if mode == "Substring" else exact_matches
            hit_count = total_sub if mode == "Substring" else len(exact_matches)

            if hit_count > 0:
                ids = [str(it.get("dbaaspId") or it.get("id")) for it in matched[:5] if it.get("dbaaspId") or it.get("id")]
                names = [str(it.get("name") or "").strip() for it in matched[:3] if str(it.get("name") or "").strip()]
                synth = next((str(it.get("synthesisType")) for it in matched if it.get("synthesisType")), "-")
                props = {
                    "DBAASP IDs": ", ".join(ids) if ids else "-",
                    "Peptide Name": ", ".join(dict.fromkeys(names)) if names else "-",
                    "Synthesis": synth,
                    "Subsequence Hits": total_sub,
                }
                return PredictionResult(
                    peptide=seq,
                    method=mode,
                    score=float(hit_count),
                    label="Found in DBAASP",
                    properties=props,
                    records=matched,
                )

            return PredictionResult(
                peptide=seq,
                method=mode,
                score=0.0,
                label="Not Found",
                properties={"DBAASP IDs": "-", "Peptide Name": "-", "Synthesis": "-", "Subsequence Hits": total_sub},
                records=[],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("DBAASP lookup failed for '%s': %s", seq, exc)
            return PredictionResult(peptide=seq, method=mode, error=str(exc))

    def predict_batch(self, sequences: list[str], *, method: str = "Exact") -> list[PredictionResult]:
        results: list[PredictionResult] = []
        for i, seq in enumerate(sequences):
            if i > 0:
                time.sleep(self.inter_request_delay)
            results.append(self.predict(seq, method=method))
        return results

    @staticmethod
    def _clean(raw: str) -> str:
        lines = [
            ln.strip()
            for ln in raw.strip().splitlines()
            if ln.strip() and not ln.strip().startswith(">")
        ]
        return "".join(lines).upper()

    @staticmethod
    def _validate(seq: str) -> Optional[str]:
        import re
        if not seq:
            return "Peptide sequence must not be empty."
        if not re.fullmatch(r"[A-Z]+", seq):
            return f"Sequence contains invalid characters: '{seq}'."
        if len(seq) < 3:
            return f"Sequence length {len(seq)} is too short (minimum 3 residues)."
        return None

    @staticmethod
    def _build_session(max_retries: int, backoff: float) -> requests.Session:
        session = requests.Session()
        session.headers.update(_DEFAULT_HEADERS)
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session


Predictor = DBAASPClient

__all__ = ["DBAASPClient", "Predictor", "PredictionResult"]
