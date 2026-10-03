"""
Predictor: submits peptide sequences to the AntiTbPred web server
and returns structured PredictionResult objects.

Target URL:   https://webs.iiitd.edu.in/raghava/antitbpred/predict3.php
Form action:  https://webs.iiitd.edu.in/raghava/antitbpred/multiple_test3.php
Result page:  https://webs.iiitd.edu.in/raghava/antitbpred/disp.php?ran=XXXXX

Server behavior:
  - POST to multiple_test3.php → meta-refresh → disp.php?ran=XXXXX
  - disp.php table is populated by Bootstrap-table JS after page load.
  - Plain requests.get() retrieves empty table cells (server-side async).
  - Playwright (Chromium headless) renders JS correctly and gets filled data.

Form fields (multipart/form-data):
  - seq          : FASTA-formatted text
  - uploadedfile : optional file (not used)
  - method       : '1'–'4'
  - thval        : SVM threshold (default '0')

Result table columns (JS-rendered):
  ID | Seq | Score | Prediction | Steric hindrance | Amphipathicity | Net Hydrogen | pI
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Union

from .models import Method, PredictionResult
from .parser import Parser

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BASE_URL   = "https://webs.iiitd.edu.in/raghava/antitbpred"
_FORM_URL   = f"{_BASE_URL}/predict3.php"


# ──────────────────────────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Submits peptide sequences to AntiTbPred via Playwright (Chromium)
    and returns PredictionResult objects.

    The server's result page uses Bootstrap-table with JS-rendered data;
    plain HTTP clients only receive empty table cells.
    Playwright ensures the full JS execution and table population.

    Parameters
    ----------
    method : Method | str
        Prediction method:
          - 'AntiTB_MD_SVM' / 'md_svm' / '1'  (default)
          - 'AntiTB_RD_SVM' / 'rd_svm' / '2'
          - 'AntiTB_MD_HYBRID' / 'md_hybrid' / '3'
          - 'AntiTB_RD_HYBRID' / 'rd_hybrid' / '4'
    threshold : float
        SVM decision threshold (-1.0 to 1.0, default: 0.0).
    timeout : float
        Page navigation timeout in milliseconds (default: 60_000).
    post_submit_wait : float
        Seconds to wait after navigation for JS to populate the table
        (default: 8.0).
    inter_request_delay : float
        Seconds to wait between successive requests in batch mode (default: 3.0).
    headless : bool
        Whether to run Chromium headlessly (default: True).

    Examples
    --------
    Single peptide:

    >>> predictor = Predictor(method='md_svm', threshold=0.0)
    >>> result = predictor.predict("NALAALAKKRQIKIW")
    >>> print(result.label, result.score)

    Batch:

    >>> results = predictor.predict_batch(["NALAALAKKRQIKIW", "EEEAAKKK"])
    >>> PredictionResult.to_csv(results, "results.csv")
    """

    def __init__(
        self,
        method: Union[Method, str] = Method.AntiTB_MD_SVM,
        *,
        threshold: float = 0.0,
        timeout: float = 60_000,
        post_submit_wait: float = 8.0,
        inter_request_delay: float = 3.0,
        headless: bool = True,
        fields: Optional[list[str]] = None,
    ) -> None:
        self.method    = Method.resolve(method) if not isinstance(method, Method) else method
        self.threshold = threshold
        self.timeout   = timeout
        self.post_submit_wait = post_submit_wait
        self.inter_request_delay = inter_request_delay
        self.headless  = headless
        self.fields = fields
        self._parser   = Parser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, peptide: str) -> PredictionResult:
        """
        Predict whether a single peptide is anti-tubercular.

        Parameters
        ----------
        peptide : str
            Amino-acid sequence (4–50 residues, uppercase letters only).

        Returns
        -------
        PredictionResult
            ``.success`` is ``False`` and ``.error`` is set on failure.
        """
        peptide = peptide.strip().upper()
        method_name = self.method.name

        validation_error = self._validate(peptide)
        if validation_error:
            logger.warning("Validation failed for '%s': %s", peptide, validation_error)
            return PredictionResult(
                peptide=peptide,
                method=method_name,
                threshold=self.threshold,
                error=validation_error,
            )

        try:
            html = self._submit(peptide)
        except Exception as exc:  # noqa: BLE001
            logger.error("Request failed for '%s': %s", peptide, exc)
            return PredictionResult(
                peptide=peptide,
                method=method_name,
                threshold=self.threshold,
                error=str(exc),
            )

        score, label, properties = self._parser.parse(html)
        result = PredictionResult(
            peptide=peptide,
            method=method_name,
            threshold=self.threshold,
            score=score,
            label=label,
            properties=properties,
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
        on_error : {'continue', 'raise'}

        Returns
        -------
        list[PredictionResult]
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
        Use Playwright to submit the form and return the fully JS-rendered
        result page HTML.

        Steps:
        1. Navigate to the form page (predict3.php).
        2. Fill in the FASTA sequence textarea.
        3. Select the prediction method radio button.
        4. Set the threshold dropdown.
        5. Click Submit → wait for navigation to disp.php?ran=XXXXX.
        6. Wait for Bootstrap-table JS to populate the table.
        7. Return page.content().
        """
        from playwright.sync_api import sync_playwright

        fasta = f">peptide\n{peptide}"
        threshold_str = str(self.threshold)
        # Normalize: 0.0 → "0", 0.5 → "0.5"
        if threshold_str.endswith(".0"):
            threshold_str = threshold_str[:-2]

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            )
            page = context.new_page()

            logger.debug("Navigating to form: %s", _FORM_URL)
            page.goto(_FORM_URL, timeout=self.timeout, wait_until="networkidle")

            # Fill FASTA sequence
            page.fill("textarea[name='seq']", fasta)
            logger.debug("Filled sequence: %s", fasta)

            # Select prediction method radio
            radio_id = f"id_radio{self.method.value}"
            page.check(f"input[id='{radio_id}']")
            logger.debug("Selected method: %s (radio: %s)", self.method.name, radio_id)

            # Set threshold dropdown (visible dropdown depends on selected method)
            # All divs contain a <select name="thval"> — select the visible one
            page.evaluate(
                f"""
                document.querySelectorAll('select[name="thval"]').forEach(sel => {{
                    for (let opt of sel.options) {{
                        if (opt.value === '{threshold_str}') {{
                            sel.value = '{threshold_str}';
                        }}
                    }}
                }});
                """
            )
            logger.debug("Set threshold: %s", threshold_str)

            # Set the checkboxes for the requested fields on the form page
            if self.fields:
                page.evaluate("""(fields) => {
                    document.querySelectorAll('input[name="field[]"]').forEach(cb => {
                        cb.checked = fields.includes(cb.value);
                    });
                }""", self.fields)
                logger.debug("Set fields: %s", self.fields)

            # Submit and wait for result page
            logger.debug("Submitting form...")
            with page.expect_navigation(timeout=self.timeout, wait_until="networkidle"):
                page.click("input[type='submit']")

            logger.debug("Result page URL: %s", page.url)

            # Wait for Bootstrap-table JS to populate the table
            time.sleep(self.post_submit_wait)

            html = page.content()
            browser.close()

        return html

    @staticmethod
    def _validate(peptide: str) -> Optional[str]:
        """Validate the peptide sequence. Returns error string or None."""
        import re
        if not peptide:
            return "Peptide sequence must not be empty."
        if not re.fullmatch(r"[A-Z]+", peptide):
            return f"Sequence contains invalid characters: '{peptide}'."
        invalid = re.search(r"[BJOUZX]", peptide)
        if invalid:
            return f"Sequence contains non-natural residue: '{invalid.group()}'."
        if len(peptide) < 4:
            return f"Sequence length {len(peptide)} < 4 (minimum)."
        if len(peptide) > 50:
            return f"Sequence length {len(peptide)} > 50 (maximum)."
        return None
