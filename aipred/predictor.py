"""
Predictor: submits peptide sequences to the AIPpred web server
and returns structured PredictionResult objects.

Target URL  : http://www.thegleelab.org/AIPpred/
Form action : http://www.thegleelab.org/cgi-bin/AIPpred/AIPpred.py  (POST)
Field name  : Sequence  (FASTA text; 5–25 amino acids)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import PredictionResult
from .parser import Parser

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BASE_URL = "http://www.thegleelab.org"
_PREDICT_URL = f"{_BASE_URL}/cgi-bin/AIPpred/AIPpred.py"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": f"{_BASE_URL}/AIPpred/",
    "Origin": _BASE_URL,
}

_MIN_LEN = 5
_MAX_LEN = 25


# ──────────────────────────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Submits peptide sequences to AIPpred and returns PredictionResult objects.

    AIPpred predicts whether a peptide is Anti-Inflammatory (AIP) using
    a Random Forest model trained with Dipeptide Composition (DPC) features.
    Accepted peptide length: 5–25 amino acids.

    Parameters
    ----------
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

    >>> predictor = Predictor()
    >>> result = predictor.predict("FLPLIGRVLSGIL")
    >>> print(result.label, result.score)

    Batch:

    >>> results = predictor.predict_batch(["FLPLIGRVLSGIL", "ACDEFGHIKLM"])
    >>> PredictionResult.to_csv(results, "aipred_results.csv")
    """

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        inter_request_delay: float = 1.0,
    ) -> None:
        self.timeout = timeout
        self.inter_request_delay = inter_request_delay

        self._parser = Parser()
        self._session = self._build_session(max_retries, retry_backoff)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, peptide: str) -> PredictionResult:
        """
        Predict whether a single peptide is anti-inflammatory.

        Parameters
        ----------
        peptide : str
            Amino-acid sequence (letters only, 5–25 residues).
            FASTA headers (lines starting with '>') are stripped automatically.

        Returns
        -------
        PredictionResult
            ``.success`` is ``False`` and ``.error`` is set on failure.
        """
        peptide = self._clean(peptide)

        validation_error = self._validate(peptide)
        if validation_error:
            logger.warning("Validation failed for '%s': %s", peptide, validation_error)
            return PredictionResult(peptide=peptide, error=validation_error)

        try:
            html = self._submit(peptide)
        except Exception as exc:  # noqa: BLE001
            logger.error("Request failed for '%s': %s", peptide, exc)
            return PredictionResult(peptide=peptide, error=str(exc))

        score, label = self._parser.parse(html)
        result = PredictionResult(
            peptide=peptide,
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
        POST the peptide to AIPpred and return the result page HTML.

        The form accepts plain sequence or FASTA format. We send a minimal
        FASTA entry so the server can identify the sequence unambiguously.
        """
        from utils.security import build_allowed_hosts, safe_request_text

        # Server requires CRLF line endings to recognize valid FASTA format
        fasta_input = f">peptide\r\n{peptide}"

        payload = {"Sequence": fasta_input}
        allowed_hosts = build_allowed_hosts(_BASE_URL)

        logger.debug("POST %s  payload=%s", _PREDICT_URL, payload)

        return safe_request_text(
            self._session,
            "POST",
            _PREDICT_URL,
            allowed_hosts=allowed_hosts,
            timeout=self.timeout,
            data=payload,
        )

    def _validate(self, peptide: str) -> Optional[str]:
        """
        Validate the peptide sequence before submission.
        Returns an error message string, or None if valid.
        """
        import re

        if not peptide:
            return "Peptide sequence must not be empty."
        if not re.fullmatch(r"[A-Z]+", peptide):
            return f"Sequence contains invalid characters: '{peptide}'."
        if len(peptide) < _MIN_LEN or len(peptide) > _MAX_LEN:
            return (
                f"Sequence length {len(peptide)} is out of range. "
                f"Accepted: {_MIN_LEN}–{_MAX_LEN} residues."
            )
        return None

    @staticmethod
    def _clean(raw: str) -> str:
        """
        Strip FASTA header lines and whitespace; return the uppercase sequence.
        """
        lines = [
            line.strip()
            for line in raw.strip().splitlines()
            if line.strip() and not line.strip().startswith(">")
        ]
        return "".join(lines).upper()

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
