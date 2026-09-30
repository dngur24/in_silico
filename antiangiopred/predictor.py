"""
Predictor: submits peptide sequences to the AntiAngioPred web server
and returns structured PredictionResult objects.

Target URL: https://webs.iiitd.edu.in/raghava/antiangiopred/predict.html
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import PredictionResult
from .parser import Parser

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BASE_URL = "https://webs.iiitd.edu.in/raghava/antiangiopred"
_PREDICT_URL = f"{_BASE_URL}/pep_test.php"  # Form action endpoint (action="pep_test.php")

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": f"{_BASE_URL}/predict.html",
    "Origin": "https://webs.iiitd.edu.in",
}


# ──────────────────────────────────────────────────────────────────────────────
# Method enum
# ──────────────────────────────────────────────────────────────────────────────

class Method(str, Enum):
    """
    Prediction methods available on the AntiAngioPred server.

    NT15
        Uses only the first 15 N-terminal residues.
        Requires peptide length ≥ 15.
    FULL_SEQ
        Uses the full peptide sequence (5–50 residues).
    """

    NT15 = "1"       # radio value sent to the server
    FULL_SEQ = "2"


# ──────────────────────────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Submits peptide sequences to AntiAngioPred and returns PredictionResult objects.

    Parameters
    ----------
    method : Method | str
        Prediction method. Use ``Method.NT15`` or ``Method.FULL_SEQ``.
        Can also be the string ``'NT15'`` or ``'FullSeq'``.
    timeout : float
        HTTP request timeout in seconds (default: 30).
    max_retries : int
        Number of times to retry on transient errors (default: 3).
    retry_backoff : float
        Exponential back-off factor for retries (default: 1.0 → waits 1, 2, 4 s).
    inter_request_delay : float
        Seconds to wait between successive requests when running batch predictions.
        Helps avoid rate-limiting (default: 1.0).

    Examples
    --------
    Single peptide:

    >>> predictor = Predictor(method=Method.FULL_SEQ)
    >>> result = predictor.predict("YCNINEVCHYARRNDKSYWL")
    >>> print(result.label, result.score)

    Batch:

    >>> results = predictor.predict_batch(["YCNINEVCHYARRNDKSYWL", "ACDEFGHIKLM"])
    >>> PredictionResult.to_csv(results, "results.csv")
    """

    def __init__(
        self,
        method: Union[Method, str] = Method.FULL_SEQ,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        inter_request_delay: float = 1.0,
    ) -> None:
        self.method = self._resolve_method(method)
        self.timeout = timeout
        self.inter_request_delay = inter_request_delay

        self._parser = Parser()
        self._session = self._build_session(max_retries, retry_backoff)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, peptide: str) -> PredictionResult:
        """
        Predict whether a single peptide is anti-angiogenic.

        Parameters
        ----------
        peptide : str
            Amino-acid sequence (letters only, 5–50 residues;
            ≥15 residues required for NT15 method).

        Returns
        -------
        PredictionResult
            ``.success`` is ``False`` and ``.error`` is set on failure.
        """
        peptide = peptide.strip().upper()
        method_name = "NT15" if self.method == Method.NT15 else "FullSeq"

        validation_error = self._validate(peptide)
        if validation_error:
            logger.warning("Validation failed for '%s': %s", peptide, validation_error)
            return PredictionResult(
                peptide=peptide,
                method=method_name,
                error=validation_error,
            )

        try:
            html = self._submit(peptide)
        except Exception as exc:  # noqa: BLE001
            logger.error("Request failed for '%s': %s", peptide, exc)
            return PredictionResult(
                peptide=peptide,
                method=method_name,
                error=str(exc),
            )

        score, label = self._parser.parse(html)
        result = PredictionResult(
            peptide=peptide,
            method=method_name,
            score=score,
            label=label,
            raw_html=html,
        )
        logger.info("Predicted '%s' → label=%s score=%s", peptide, label, score)
        return result

    def predict_batch(
        self,
        peptides: list[str],
        *,
        on_error: str = "continue",
    ) -> list[PredictionResult]:
        """
        Predict a list of peptides sequentially.

        Parameters
        ----------
        peptides : list[str]
            List of amino-acid sequences.
        on_error : {'continue', 'raise'}
            What to do on a failed prediction.
            ``'continue'`` (default) stores the error in the result and moves on.
            ``'raise'`` re-raises the exception immediately.

        Returns
        -------
        list[PredictionResult]
            One result per input peptide, in the same order.
        """
        results: list[PredictionResult] = []
        for i, peptide in enumerate(peptides):
            if i > 0:
                time.sleep(self.inter_request_delay)

            result = self.predict(peptide)

            if not result.success and on_error == "raise":
                raise RuntimeError(
                    f"Prediction failed for '{peptide}': {result.error}"
                )

            results.append(result)
            logger.debug(
                "[%d/%d] %s → %s", i + 1, len(peptides), peptide, result.label
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit(self, peptide: str) -> str:
        """
        Submit the peptide and return the final result page HTML.

        The server uses a 2-step pattern:
          1. POST → pep_test.php  (returns HTML with meta-refresh redirect)
          2. GET  → submitfreq.php?ran=XXXXX  (actual result page)
        """
        import re as _re

        payload = {
            "seq": peptide,
            "method": self.method.value,
            "submit": "Submit",
        }
        logger.debug("POST %s  payload=%s", _PREDICT_URL, payload)

        # ── Step 1: POST ─────────────────────────────────────────────────
        response = self._session.post(
            _PREDICT_URL,
            data=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        # ── Step 2: Follow meta-refresh redirect ─────────────────────────
        #   <meta http-equiv='refresh' content='0;url=submitfreq.php?ran=XXXXX' />
        m = _re.search(r"content=['\"]0;url=([^'\"]+)['\"]", response.text, _re.IGNORECASE)
        if not m:
            logger.warning("No meta-refresh redirect found in response; returning raw page.")
            return response.text

        redirect_path = m.group(1)
        # Build absolute URL
        if redirect_path.startswith("http"):
            result_url = redirect_path
        else:
            result_url = f"{_BASE_URL}/{redirect_path}"

        logger.debug("GET result page: %s", result_url)
        result_response = self._session.get(result_url, timeout=self.timeout)
        result_response.raise_for_status()
        return result_response.text

    def _validate(self, peptide: str) -> Optional[str]:
        """
        Replicate the client-side JS validation from the prediction page.
        Returns an error message string, or None if valid.
        """
        import re

        if not peptide:
            return "Peptide sequence must not be empty."
        if not re.fullmatch(r"[A-Z]+", peptide):
            return f"Sequence contains invalid characters: '{peptide}'."
        if len(peptide) < 5:
            return (
                f"Sequence length {len(peptide)} is too short. "
                "Minimum: 5 residues."
            )
        if self.method == Method.NT15 and len(peptide) < 15:
            return (
                f"NT15 method requires ≥15 residues; got {len(peptide)}."
            )
        return None

    @staticmethod
    def _build_session(max_retries: int, backoff: float) -> requests.Session:
        """Create a requests Session with retry logic."""
        session = requests.Session()
        session.headers.update(_DEFAULT_HEADERS)

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    @staticmethod
    def _resolve_method(method: Union[Method, str]) -> Method:
        """Accept Method enum, its value string, or friendly name."""
        if isinstance(method, Method):
            return method
        mapping = {
            "NT15": Method.NT15,
            "nt15": Method.NT15,
            "1": Method.NT15,
            "FULLSEQ": Method.FULL_SEQ,
            "fullseq": Method.FULL_SEQ,
            "FullSeq": Method.FULL_SEQ,
            "2": Method.FULL_SEQ,
        }
        resolved = mapping.get(str(method))
        if resolved is None:
            raise ValueError(
                f"Unknown method '{method}'. Use Method.NT15 or Method.FULL_SEQ."
            )
        return resolved
