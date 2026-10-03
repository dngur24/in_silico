"""
Data models for AntiTbPred results.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import csv
import json
from pathlib import Path


class Method(str, Enum):
    """
    Prediction methods available on the AntiTbPred server.

    AntiTB_MD_SVM
        SVM ensemble method trained on main dataset (default).
    AntiTB_RD_SVM
        SVM ensemble method trained on random peptide dataset.
    AntiTB_MD_HYBRID
        Hybrid (SVM + motif) method trained on main dataset.
    AntiTB_RD_HYBRID
        Hybrid (SVM + motif) method trained on random peptide dataset.
    """

    AntiTB_MD_SVM    = "1"
    AntiTB_RD_SVM    = "2"
    AntiTB_MD_HYBRID = "3"
    AntiTB_RD_HYBRID = "4"

    _ALIAS_MAP: dict[str, str] = {}  # populated below

    @classmethod
    def resolve(cls, value: str) -> "Method":
        """Resolve alias or raw value to a Method enum member."""
        mapping = {
            # friendly names
            "antitb_md_svm":    cls.AntiTB_MD_SVM,
            "md_svm":           cls.AntiTB_MD_SVM,
            "1":                cls.AntiTB_MD_SVM,
            "antitb_rd_svm":    cls.AntiTB_RD_SVM,
            "rd_svm":           cls.AntiTB_RD_SVM,
            "2":                cls.AntiTB_RD_SVM,
            "antitb_md_hybrid": cls.AntiTB_MD_HYBRID,
            "md_hybrid":        cls.AntiTB_MD_HYBRID,
            "hybrid":           cls.AntiTB_MD_HYBRID,
            "3":                cls.AntiTB_MD_HYBRID,
            "antitb_rd_hybrid": cls.AntiTB_RD_HYBRID,
            "rd_hybrid":        cls.AntiTB_RD_HYBRID,
            "4":                cls.AntiTB_RD_HYBRID,
        }
        resolved = mapping.get(str(value).lower())
        if resolved is None:
            raise ValueError(
                f"Unknown method '{value}'. "
                f"Valid options: {list(mapping.keys())}"
            )
        return resolved


@dataclass
class PredictionResult:
    """
    Standardized result from one AntiTbPred prediction.

    Attributes
    ----------
    peptide : str
        Input peptide sequence (upper-cased).
    method : str
        Prediction method used (e.g. 'AntiTB_MD_SVM').
    threshold : float
        SVM threshold used for classification (default: 0.0).
    score : Optional[float]
        SVM score returned by the server (None if not parsed).
    label : Optional[str]
        Classification label: 'AntiTB' or 'Non-AntiTB'.
    raw_html : Optional[str]
        Full HTML of the result page (for debugging; not exported to CSV).
    error : Optional[str]
        Error message if the prediction failed.
    """

    peptide: str
    method: str
    threshold: float = 0.0
    score: Optional[float] = None
    label: Optional[str] = None
    properties : Optional[dict[str, Any]] = None
    raw_html: Optional[str] = field(default=None, repr=False)
    error: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def success(self) -> bool:
        """True when the prediction completed without errors."""
        return self.error is None

    def to_dict(self, include_raw_html: bool = False) -> dict:
        """Return a plain dictionary representation."""
        d = {
            "peptide":    self.peptide,
            "method":     self.method,
            "threshold":  self.threshold,
            "score":      self.score,
            "label":      self.label,
            "properties": self.properties,
            "error":      self.error,
        }
        if include_raw_html:
            d["raw_html"] = self.raw_html
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string (without raw_html)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    # ------------------------------------------------------------------
    # Batch helpers (class-level)
    # ------------------------------------------------------------------

    @classmethod
    def to_csv(
        cls,
        results: list["PredictionResult"],
        path: str | Path,
        *,
        encoding: str = "utf-8-sig",  # BOM for Excel compatibility
    ) -> Path:
        """
        Write a list of PredictionResult objects to a CSV file.

        Parameters
        ----------
        results : list[PredictionResult]
        path : str or Path  – destination file path
        encoding : str

        Returns
        -------
        Path  – resolved path to the written file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        properties_keys = []
        for r in results:
            if r.properties:
                for k in r.properties.keys():
                    if k not in properties_keys:
                        properties_keys.append(k)

        fieldnames = ["peptide", "method", "threshold", "score", "label"] + properties_keys +["error"]
        with path.open("w", newline="", encoding=encoding) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = r.to_dict()
                if r.properties:
                    row.update(r.properties)
                writer.writerow(row)

        return path.resolve()
