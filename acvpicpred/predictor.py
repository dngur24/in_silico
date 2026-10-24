"""
Predictor: submits peptide sequences to the ACVPICPred web server
and returns structured PredictionResult objects.

Target URL: https://i.uestc.edu.cn/acvpICPred/main/Main.php

Form fields (multipart/form-data POST):
    pep         – peptide sequence (natural amino acids only)
    upload_file – PDB structure file (.pdb)
    Virus       – virus type (e.g. "SARS-CoV-2")
    Assay       – experiment type (e.g. "cell-cell fusion")
    Value_Type  – inhibition value type (IC50 / EC50 / IC90)
    Unit        – unit type (uM / ug/ML)
    submit      – submit button value ("Predict")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import (
    AssayType,
    PredictionResult,
    UnitType,
    ValueType,
    VirusType,
)
from .parser import Parser

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BASE_URL    = "https://i.uestc.edu.cn/acvpICPred"
_PREDICT_URL = f"{_BASE_URL}/main/Main.php"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": _PREDICT_URL,
    "Origin":  "https://i.uestc.edu.cn",
}


# ──────────────────────────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Submits peptide sequences to ACVPICPred and returns PredictionResult objects.

    Parameters
    ----------
    virus : VirusType | str
        Virus type to predict against (default: SARS-CoV-2).
    assay : AssayType | str
        Experiment type (default: cell-cell fusion).
    value_type : ValueType | str
        Inhibition value type – IC50, EC50, or IC90 (default: IC50).
    unit : UnitType | str
        Unit for the inhibition value – uM or ug/ML (default: uM).
    timeout : float
        HTTP request timeout in seconds (default: 60).
    max_retries : int
        Number of times to retry on transient HTTP errors (default: 3).
    retry_backoff : float
        Exponential back-off factor for retries (default: 1.0 → waits 1, 2, 4 s).
    inter_request_delay : float
        Seconds to wait between successive requests in batch mode (default: 2.0).

    Examples
    --------
    Single prediction:

    >>> predictor = Predictor(virus=VirusType.SARS_CoV_2)
    >>> result = predictor.predict("ACDEFGHIKLM", pdb_path="peptide.pdb")
    >>> print(result.label, result.predicted_value)

    Batch prediction (same PDB for all, or one PDB per peptide):

    >>> items = [
    ...     ("ACDEFGHIKLM", "pep1.pdb"),
    ...     ("YCNINEVCHYA", "pep2.pdb"),
    ... ]
    >>> results = predictor.predict_batch(items)
    >>> PredictionResult.to_csv(results, "results.csv")
    """

    def __init__(
        self,
        virus:      Union[VirusType, str]  = VirusType.SARS_CoV_2,
        assay:      Union[AssayType, str]  = AssayType.CELL_CELL_FUSION,
        value_type: Union[ValueType, str]  = ValueType.IC50,
        unit:       Union[UnitType, str]   = UnitType.uM,
        *,
        timeout:              float = 60.0,
        max_retries:          int   = 3,
        retry_backoff:        float = 1.0,
        inter_request_delay:  float = 2.0,
    ) -> None:
        self.virus      = self._coerce(virus,      VirusType)
        self.assay      = self._coerce(assay,      AssayType)
        self.value_type = self._coerce(value_type, ValueType)
        self.unit       = self._coerce(unit,       UnitType)

        self.timeout              = timeout
        self.inter_request_delay  = inter_request_delay

        self._parser  = Parser()
        self._session = self._build_session(max_retries, retry_backoff)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        peptide:  str,
        pdb_path: Optional[Union[str, Path]] = None,
    ) -> PredictionResult:
        """
        Predict the anti-coronavirus activity of a single peptide.

        Parameters
        ----------
        peptide : str
            Amino-acid sequence (natural amino acids only).
        pdb_path : str | Path | None
            Path to the corresponding PDB structure file. When ``None``,
            an idealized 3D peptide structure (.pdb) is auto-generated
            in a temporary file and cleaned up after submission.

        Returns
        -------
        PredictionResult
            ``.success`` is ``False`` and ``.error`` is set on failure.
        """
        from .pdb_generator import write_temp_peptide_pdb

        peptide = peptide.strip().upper()
        temp_pdb: Optional[Path] = None

        try:
            if pdb_path is None:
                try:
                    temp_pdb = write_temp_peptide_pdb(peptide)
                    resolved_pdb = temp_pdb
                except Exception as exc:  # noqa: BLE001
                    return self._error_result(peptide, Path("auto.pdb"), str(exc))
            else:
                resolved_pdb = Path(pdb_path)

            # ── Validation ────────────────────────────────────────────────────
            validation_error = self._validate(peptide, resolved_pdb)
            if validation_error:
                logger.warning("Validation failed for '%s': %s", peptide, validation_error)
                return self._error_result(peptide, resolved_pdb, validation_error)

            # ── Submit ────────────────────────────────────────────────────────
            try:
                html = self._submit(peptide, resolved_pdb)
            except Exception as exc:  # noqa: BLE001
                logger.error("Request failed for '%s': %s", peptide, exc)
                return self._error_result(peptide, resolved_pdb, str(exc))

            # ── Parse ─────────────────────────────────────────────────────────
            predicted_value, label = self._parser.parse(html)
            result = PredictionResult(
                peptide=peptide,
                pdb_path="auto-generated" if temp_pdb is not None else str(resolved_pdb),
                virus=self.virus.value,
                assay=self.assay.value,
                value_type=self.value_type.value,
                unit=self.unit.value,
                predicted_value=predicted_value,
                label=label,
                raw_html=html,
            )
            logger.info(
                "Predicted '%s' → label=%s value=%s", peptide, label, predicted_value
            )
            return result
        finally:
            if temp_pdb is not None:
                temp_pdb.unlink(missing_ok=True)

    def predict_batch(
        self,
        items: list[Union[str, tuple[str, Optional[Union[str, Path]]]]],
        *,
        on_error: str = "continue",
    ) -> list[PredictionResult]:
        """
        Predict a list of peptide sequences (auto-generating PDBs) or
        ``(peptide, pdb_path)`` pairs sequentially.
        """
        results: list[PredictionResult] = []

        for i, item in enumerate(items):
            if i > 0:
                time.sleep(self.inter_request_delay)

            if isinstance(item, tuple):
                peptide, pdb_path = item
            else:
                peptide, pdb_path = item, None

            result = self.predict(peptide, pdb_path)

            if not result.success and on_error == "raise":
                raise RuntimeError(
                    f"Prediction failed for '{peptide}': {result.error}"
                )

            results.append(result)
            logger.debug(
                "[%d/%d] %s → %s", i + 1, len(items), peptide, result.label
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit(self, peptide: str, pdb_path: Path) -> str:
        """
        Submit the multipart/form-data POST and return the result-page HTML.

        The server may use a meta-refresh redirect; we follow it if present.
        """
        import re as _re
        from utils.security import (
            build_allowed_hosts,
            resolve_safe_redirect,
            safe_request_text,
        )

        allowed_hosts = build_allowed_hosts(_BASE_URL)
        with pdb_path.open("rb") as pdb_file:
            files = {
                "upload_file": (pdb_path.name, pdb_file, "chemical/x-pdb"),
            }
            data = {
                "pep":        peptide,
                "Virus":      self.virus.value,
                "Assay":      self.assay.value,
                "Value_Type": self.value_type.value,
                "Unit":       self.unit.value,
                "submit":     "Predict",
            }

            logger.debug("POST %s  peptide=%s  pdb=%s", _PREDICT_URL, peptide, pdb_path.name)
            logger.debug("Params: virus=%s assay=%s value_type=%s unit=%s",
                         self.virus.value, self.assay.value,
                         self.value_type.value, self.unit.value)

            first_html = safe_request_text(
                self._session,
                "POST",
                _PREDICT_URL,
                allowed_hosts=allowed_hosts,
                timeout=self.timeout,
                data=data,
                files=files,
            )

        # ── Follow validated meta-refresh redirect if present ─────────────────
        m = _re.search(
            r"content=['\"]0;url=([^'\"]+)['\"]",
            first_html,
            _re.IGNORECASE,
        )
        if not m:
            logger.debug("No meta-refresh; returning direct response.")
            return first_html

        redirect_path = m.group(1)
        result_url = resolve_safe_redirect(redirect_path, _BASE_URL, allowed_hosts)

        logger.debug("GET result page: %s", result_url)
        return safe_request_text(
            self._session,
            "GET",
            result_url,
            allowed_hosts=allowed_hosts,
            timeout=self.timeout,
        )

    def _validate(self, peptide: str, pdb_path: Path) -> Optional[str]:
        """
        Validate inputs before submitting.
        Returns an error message string, or None if valid.
        """
        import re

        if not peptide:
            return "Peptide sequence must not be empty."
        if not re.fullmatch(r"[A-Z]+", peptide):
            return f"Sequence contains invalid characters: '{peptide}'."
        if not pdb_path.exists():
            return f"PDB file not found: '{pdb_path}'."
        if pdb_path.suffix.lower() != ".pdb":
            return f"File must have .pdb extension, got: '{pdb_path.suffix}'."
        return None

    def _error_result(
        self, peptide: str, pdb_path: Path, error: str
    ) -> PredictionResult:
        """Convenience builder for failed results."""
        return PredictionResult(
            peptide=peptide,
            pdb_path=str(pdb_path),
            virus=self.virus.value,
            assay=self.assay.value,
            value_type=self.value_type.value,
            unit=self.unit.value,
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
        session.mount("http://", adapter)
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
