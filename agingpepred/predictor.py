"""
Predictor: submits peptide sequences to the AagingPEPred web server
and returns structured PredictionResult objects.

Target form page: https://project.iith.ac.in/cgntlab/aagingpepred/pep_pred.php
Form action:      https://project.iith.ac.in/cgntlab/aagingpepred/process_pep_pred.php

Form fields (enctype="multipart/form-data"):
  - sequences : FASTA-formatted text (e.g. ">id\\nSEQUENCE")
  - files[]   : optional file upload (not used here)
  - model     : one of the four model values (see models.Model)

The server blocks plain HTTP requests (403 Forbidden) without a real browser
context.  This implementation uses Playwright (Chromium headless) to submit
the form as a real browser would.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Union

from .models import Model, PredictionResult
from .parser import Parser

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BASE_URL = "https://project.iith.ac.in/cgntlab/aagingpepred"
_FORM_URL = f"{_BASE_URL}/pep_pred.php"
_ACTION_URL = f"{_BASE_URL}/process_pep_pred.php"


# ──────────────────────────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Submits peptide sequences to AagingPEPred via Playwright (Chromium)
    and returns PredictionResult objects.

    Parameters
    ----------
    model : str
        Prediction model to use. One of:
          - 'stochastic' / 'all_features_randomfeatures'  (default, recommended)
          - 'literature' / 'all_features_swissprot'
          - 'targeted'   / 'rfe_features_randomfeatures'
          - 'high_precision' / 'rfe_features_swissprot'
    timeout : float
        Page load / navigation timeout in milliseconds (default: 60_000).
    inter_request_delay : float
        Seconds to wait between successive requests in batch mode (default: 2.0).
    headless : bool
        Whether to run Chromium headlessly (default: True).

    Examples
    --------
    Single peptide:

    >>> predictor = Predictor(model='stochastic')
    >>> result = predictor.predict("YPSKPDNPGEDAPAEDMARYYSALRHYINLITRQRY")
    >>> print(result.label, result.score)

    Batch:

    >>> results = predictor.predict_batch(["PEPTIDE1", "PEPTIDE2"])
    >>> PredictionResult.to_csv(results, "results.csv")
    """

    def __init__(
        self,
        model: str = Model.STOCHASTIC,
        *,
        timeout: float = 60_000,
        inter_request_delay: float = 2.0,
        headless: bool = True,
    ) -> None:
        self.model = Model.resolve(model)
        self.timeout = timeout
        self.inter_request_delay = inter_request_delay
        self.headless = headless
        self._parser = Parser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, peptide: str) -> PredictionResult:
        """
        Predict whether a single peptide has anti-aging activity.

        Parameters
        ----------
        peptide : str
            Amino-acid sequence (5–50 residues, letters only).

        Returns
        -------
        PredictionResult
            ``.success`` is ``False`` and ``.error`` is set on failure.
        """
        peptide = peptide.strip().upper()

        validation_error = self._validate(peptide)
        if validation_error:
            logger.warning("Validation failed for '%s': %s", peptide, validation_error)
            return PredictionResult(
                peptide=peptide,
                model=self.model,
                error=validation_error,
            )

        try:
            html = self._submit(peptide)
        except Exception as exc:  # noqa: BLE001
            logger.error("Request failed for '%s': %s", peptide, exc)
            return PredictionResult(
                peptide=peptide,
                model=self.model,
                error=str(exc),
            )

        score, label = self._parser.parse(html)
        result = PredictionResult(
            peptide=peptide,
            model=self.model,
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
            ``'continue'`` (default) stores the error and moves on.
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
        Use Playwright to open the form page, fill in the peptide sequence,
        select the model, submit, and return the result page HTML.

        The server requires a real browser context (403 on plain HTTP requests).
        """
        from playwright.sync_api import sync_playwright
        from utils.security import (
            DEFAULT_MAX_RESPONSE_BYTES,
            build_allowed_hosts,
            create_hardened_playwright_page,
            validate_safe_url,
        )

        allowed_hosts = build_allowed_hosts(_BASE_URL)
        fasta = f">peptide\n{peptide}"

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self.headless)
            try:
                _context, page = create_hardened_playwright_page(
                    browser,
                    allowed_hosts=allowed_hosts,
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                )

                logger.debug("Navigating to form page: %s", _FORM_URL)
                page.goto(_FORM_URL, timeout=self.timeout, wait_until="networkidle")
                validate_safe_url(page.url, allowed_hosts)

                # Fill the textarea with FASTA sequence
                page.fill("textarea[name='sequences']", fasta)
                logger.debug("Filled sequences textarea with: %s", fasta)

                # Select the model radio button
                page.check(f"input[name='model'][value='{self.model}']")
                logger.debug("Selected model: %s", self.model)

                # Submit the form and wait for result page to load
                logger.debug("Submitting form to %s", _ACTION_URL)
                with page.expect_navigation(timeout=self.timeout, wait_until="networkidle"):
                    page.click("button[type='submit']")

                validate_safe_url(page.url, allowed_hosts)
                result_html = page.content()
                if len(result_html.encode("utf-8", errors="ignore")) > DEFAULT_MAX_RESPONSE_BYTES:
                    raise ValueError(
                        f"Result page exceeded safety limit ({DEFAULT_MAX_RESPONSE_BYTES} bytes)"
                    )
                logger.debug("Result URL: %s | HTML length: %d", page.url, len(result_html))
            finally:
                browser.close()

        return result_html

    @staticmethod
    def _validate(peptide: str) -> Optional[str]:
        """
        Validate the peptide sequence before submission.
        Returns an error message string, or None if valid.
        """
        import re

        if not peptide:
            return "Peptide sequence must not be empty."
        if not re.fullmatch(r"[A-Z]+", peptide):
            return f"Sequence contains invalid characters: '{peptide}'."
        if len(peptide) < 5 or len(peptide) > 50:
            return (
                f"Sequence length {len(peptide)} is out of range. "
                "Accepted: 5–50 residues."
            )
        return None
