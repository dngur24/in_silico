"""
Predictor: submits peptide sequences to the AntiCP 2.0 web server
and returns structured PredictionResult objects.

Target URL : https://webs.iiitd.edu.in/raghava/anticp2/predict.php
POST action: https://webs.iiitd.edu.in/raghava/anticp2/multiple_test3.php

Form fields (multipart/form-data POST):
    seq          – peptide sequences in FASTA format (textarea)
    uploadedfile – optional FASTA file upload (FILE)
    method       – prediction method radio button (1 / 2 / 3 / 4)
    thval        – score threshold select (0.0 – 1.0)
    field[]      – optional extra output columns (checkboxes 4–13)
    submit       – submit button value ("Submit")
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import (
    PredictionMethod,
    PredictionResult,
    ThresholdValue,
)
from .parser import Parser

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BASE_URL    = "https://webs.iiitd.edu.in/raghava/anticp2"
_PREDICT_URL = f"{_BASE_URL}/predict.php"
_SUBMIT_URL  = f"{_BASE_URL}/multiple_test3.php"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": _PREDICT_URL,
    "Origin":  "https://webs.iiitd.edu.in",
}


# ──────────────────────────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Submits peptide sequences to AntiCP 2.0 and returns PredictionResult objects.

    Parameters
    ----------
    method : PredictionMethod | str
        Prediction method to use (default: ML).
        - ``"1"`` / ``PredictionMethod.ML``     → Machine Learning
        - ``"2"`` / ``PredictionMethod.AAC``    → Amino Acid Composition
        - ``"3"`` / ``PredictionMethod.DPC``    → Dipeptide Composition
        - ``"4"`` / ``PredictionMethod.HYBRID`` → Hybrid (AAC + DPC)
    threshold : ThresholdValue | str
        Score threshold for anti-cancer classification (default: 0.5).
        Peptides with ML score >= threshold are labelled "Anti-Cancer Peptide".
    timeout : float
        HTTP request timeout in seconds (default: 120).
    max_retries : int
        Number of times to retry on transient HTTP errors (default: 3).
    retry_backoff : float
        Exponential back-off factor for retries (default: 1.0 → waits 1, 2, 4 s).
    inter_request_delay : float
        Seconds to wait between successive requests in batch mode (default: 2.0).

    Examples
    --------
    Single prediction::

        predictor = Predictor()
        results = predictor.predict("CGESKVWKIPCVTSIFNCK", name="seq1")
        for r in results:
            print(r.peptide_name, r.score, r.label)

    Batch prediction::

        items = [
            ("seq1", "CGESKVWKIPCVTSIFNCK"),
            ("seq2", "ICRLPGCAGNILNTISCKIF"),
        ]
        results = predictor.predict_batch(items)
        PredictionResult.to_csv(results, "results.csv")

    FASTA multi-sequence (single request)::

        fasta = ">seq1\\nCGESKVWKIPCVTSIFNCK\\n>seq2\\nICRLPGCAGNILNTISCKIF"
        results = predictor.predict_fasta(fasta)
    """

    def __init__(
        self,
        method:    Union[PredictionMethod, str] = PredictionMethod.ML,
        threshold: Union[ThresholdValue,   str] = ThresholdValue.T_0_5,
        *,
        timeout:              float = 120.0,
        max_retries:          int   = 3,
        retry_backoff:        float = 1.0,
        inter_request_delay:  float = 2.0,
    ) -> None:
        self.method    = self._coerce(method,    PredictionMethod)
        self.threshold = self._coerce(threshold, ThresholdValue)

        self.timeout             = timeout
        self.inter_request_delay = inter_request_delay

        self._parser  = Parser()
        self._session = self._build_session(max_retries, retry_backoff)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        peptide: str,
        name: str = "seq1",
    ) -> list[PredictionResult]:
        """
        Predict the anti-cancer activity of a single peptide.

        The sequence is wrapped in minimal FASTA format before submission.

        Parameters
        ----------
        peptide : str
            Amino-acid sequence (natural amino acids only).
        name : str
            Sequence identifier used in the FASTA header (default: ``"seq1"``).

        Returns
        -------
        list[PredictionResult]
            Typically contains one element.  On failure, returns a single
            result with ``success=False`` and an ``error`` message.
        """
        peptide = peptide.strip().upper()

        # ── Validation ────────────────────────────────────────────────────────
        if error := self._validate_sequence(peptide):
            logger.warning("Validation failed for '%s': %s", peptide, error)
            return [self._error_result(name, peptide, error)]

        fasta = f">{name}\n{peptide}\n"
        return self._submit_and_parse(fasta, fallback_name=name, fallback_seq=peptide)

    def predict_fasta(self, fasta: str) -> list[PredictionResult]:
        """
        Predict anti-cancer activity for one or more sequences in FASTA format.

        All sequences are submitted in a single HTTP request, which is the most
        efficient approach for bulk predictions.

        Parameters
        ----------
        fasta : str
            Multi-sequence FASTA string.

        Returns
        -------
        list[PredictionResult]
            One result per sequence found in the server's response table.
        """
        fasta = fasta.strip()
        if not fasta:
            return [self._error_result("", "", "FASTA input must not be empty.")]

        # Extract first sequence name/seq as fallback for error results
        lines = fasta.splitlines()
        fallback_name = lines[0].lstrip(">").strip() if lines else "seq1"
        fallback_seq  = ""
        for line in lines[1:]:
            if not line.startswith(">"):
                fallback_seq += line.strip()
            else:
                break

        return self._submit_and_parse(
            fasta,
            fallback_name=fallback_name,
            fallback_seq=fallback_seq,
        )

    def predict_batch(
        self,
        items: list[tuple[str, str]],
        *,
        on_error: str = "continue",
    ) -> list[PredictionResult]:
        """
        Predict a list of (name, peptide) pairs sequentially, one request each.

        For large batches, consider using :meth:`predict_fasta` to submit all
        sequences in a single request instead.

        Parameters
        ----------
        items : list of (name, peptide_sequence) tuples
        on_error : {'continue', 'raise'}
            ``'continue'`` (default) stores the error and moves on.
            ``'raise'`` re-raises the first error immediately.

        Returns
        -------
        list[PredictionResult]
            All results in order, including any error results.
        """
        all_results: list[PredictionResult] = []

        normalized_items: list[tuple[str, str]] = []
        for i, item in enumerate(items):
            if isinstance(item, str):
                normalized_items.append((f"seq{i+1}", item))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                normalized_items.append((str(item[0]), str(item[1])))
            else:
                normalized_items.append((f"seq{i+1}", str(item)))

        for i, (name, peptide) in enumerate(normalized_items):
            if i > 0:
                time.sleep(self.inter_request_delay)

            batch_results = self.predict(peptide, name=name)

            for result in batch_results:
                if not result.success and on_error == "raise":
                    raise RuntimeError(
                        f"Prediction failed for '{name}' / '{peptide}': "
                        f"{result.error}"
                    )

            all_results.extend(batch_results)
            logger.debug(
                "[%d/%d] %s → %s results",
                i + 1, len(items), name, len(batch_results),
            )

        return all_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit_and_parse(
        self,
        fasta: str,
        *,
        fallback_name: str,
        fallback_seq: str,
    ) -> list[PredictionResult]:
        """Submit FASTA to the server and parse the response."""
        try:
            html = self._submit(fasta)
        except Exception as exc:  # noqa: BLE001
            logger.error("Request failed: %s", exc)
            return [self._error_result(fallback_name, fallback_seq, str(exc))]

        results = self._parser.parse(
            html,
            peptide_name=fallback_name,
            peptide=fallback_seq,
            method=self.method.name,
            threshold=self.threshold.value,
        )

        # Attach raw HTML to every result for debugging
        for r in results:
            r.raw_html = html

        logger.info(
            "Parsed %d result(s) from server response.", len(results)
        )
        return results

    def _submit(self, fasta: str) -> str:
        """
        Submit the FASTA sequence(s) via POST and return the result-page HTML.

        The server may issue a meta-refresh redirect; we follow it if present.
        """
        import re as _re

        data = {
            "seq":    fasta,
            "method": self.method.value,
            "thval":  self.threshold.value,
            "submit": "Submit",
        }

        logger.debug(
            "POST %s  method=%s  threshold=%s",
            _SUBMIT_URL, self.method.name, self.threshold.value,
        )

        response = self._session.post(
            _SUBMIT_URL,
            data=data,
            timeout=self.timeout,
        )
        response.raise_for_status()

        # ── Follow meta-refresh redirect if present ───────────────────────────
        m = _re.search(
            r"content=['\"]0;url=([^'\"]+)['\"]",
            response.text,
            _re.IGNORECASE,
        )
        if not m:
            logger.debug("No meta-refresh; returning direct response.")
            return response.text

        redirect_path = m.group(1)
        result_url = (
            redirect_path
            if redirect_path.startswith("http")
            else f"{_BASE_URL}/{redirect_path.lstrip('/')}"
        )

        logger.debug("GET result page: %s", result_url)
        result_response = self._session.get(result_url, timeout=self.timeout)
        result_response.raise_for_status()
        return result_response.text

    def _validate_sequence(self, peptide: str) -> Optional[str]:
        """
        Validate a single peptide sequence.
        Returns an error string, or ``None`` if valid.
        """
        import re
        if not peptide:
            return "Peptide sequence must not be empty."
        if not re.fullmatch(r"[A-Z]+", peptide):
            return (
                f"Sequence contains invalid characters: '{peptide}'. "
                "Only uppercase amino-acid letters are allowed."
            )
        return None

    def _error_result(
        self, name: str, peptide: str, error: str
    ) -> PredictionResult:
        """Convenience builder for failed results."""
        return PredictionResult(
            peptide_name=name,
            peptide=peptide,
            method=self.method.name,
            threshold=self.threshold.value,
            error=error,
        )

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
        session.mount("http://",  adapter)
        return session

    @staticmethod
    def _coerce(value, enum_cls):
        """Accept an enum member or its .value string."""
        if isinstance(value, enum_cls):
            return value
        try:
            return enum_cls(value)
        except ValueError:
            valid = [e.value for e in enum_cls]
            raise ValueError(
                f"Invalid value '{value}' for {enum_cls.__name__}. "
                f"Valid options: {valid}"
            ) from None
