"""
Data models for AntiCP 2.0 results.

Target: https://webs.iiitd.edu.in/raghava/anticp2/predict.php
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Parameter Enums  (mirror <select> / <radio> options from the HTML form)
# ──────────────────────────────────────────────────────────────────────────────

class PredictionMethod(str, Enum):
    """
    Prediction method options available on the AntiCP 2.0 form.

    1 = Model 1 (Main dataset)
    2 = Model 2 (Alternate dataset)
    3 = Hybrid_DS1
    4 = Hybrid_DS2
    """
    MODEL_1    = "1"
    MODEL_2    = "2"
    HYBRID_DS1 = "3"
    HYBRID_DS2 = "4"

    # Backward-compatible aliases
    ML     = "1"
    AAC    = "2"
    DPC    = "3"
    HYBRID = "4"

    @property
    def display_name(self) -> str:
        return {
            "1": "Model 1",
            "2": "Model 2",
            "3": "Hybrid_DS1",
            "4": "Hybrid_DS2",
        }[self.value]

    @classmethod
    def resolve(cls, value: "PredictionMethod | str | None") -> "PredictionMethod":
        if isinstance(value, cls):
            return value
        if value is None or str(value).strip() == "":
            return cls.MODEL_1
        raw = str(value).strip()
        alias_map = {
            "1": cls.MODEL_1,
            "model 1": cls.MODEL_1,
            "model_1": cls.MODEL_1,
            "model1": cls.MODEL_1,
            "ml": cls.MODEL_1,
            "2": cls.MODEL_2,
            "model 2": cls.MODEL_2,
            "model_2": cls.MODEL_2,
            "model2": cls.MODEL_2,
            "aac": cls.MODEL_2,
            "3": cls.HYBRID_DS1,
            "hybrid_ds1": cls.HYBRID_DS1,
            "hybrid ds1": cls.HYBRID_DS1,
            "dpc": cls.HYBRID_DS1,
            "4": cls.HYBRID_DS2,
            "hybrid_ds2": cls.HYBRID_DS2,
            "hybrid ds2": cls.HYBRID_DS2,
            "hybrid": cls.HYBRID_DS2,
        }
        resolved = alias_map.get(raw.lower())
        if resolved is None:
            raise ValueError(
                f"Invalid AntiCP 2.0 method '{value}'. "
                "Valid options: 'Model 1', 'Model 2', 'Hybrid_DS1', 'Hybrid_DS2'."
            )
        return resolved


class ThresholdValue(str, Enum):
    """
    Score threshold options for classification (anti-cancer or non-anti-cancer).
    A peptide is predicted as anti-cancer if its score >= threshold.
    """
    T_0_0  = "0.0"
    T_0_1  = "0.1"
    T_0_2  = "0.2"
    T_0_3  = "0.3"
    T_0_4  = "0.4"
    T_0_45 = "0.45"
    T_0_5  = "0.5"
    T_0_6  = "0.6"
    T_0_7  = "0.7"
    T_0_8  = "0.8"
    T_0_9  = "0.9"
    T_1_0  = "1.0"

    @classmethod
    def resolve(cls, value: "ThresholdValue | float | str | None") -> "ThresholdValue":
        if isinstance(value, cls):
            return value
        if value is None or str(value).strip() == "":
            return cls.T_0_45
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid threshold '{value}'.") from exc
        if abs(num - 0.45) < 1e-6:
            return cls.T_0_45
        formatted = f"{round(num, 1):.1f}"
        for member in cls:
            if member.value == formatted:
                return member
        valid = [e.value for e in cls]
        raise ValueError(f"Invalid threshold '{value}'. Valid options: {valid}")


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Standardized result from one AntiCP 2.0 prediction (single peptide row).

    Attributes
    ----------
    peptide_name : str
        Sequence ID / name from the FASTA header (e.g. ``"seq1"``).
    peptide : str
        Amino-acid sequence (upper-cased).
    method : str
        Prediction method used (ML / AAC / DPC / Hybrid).
    threshold : str
        Score threshold used for classification.
    score : Optional[float]
        Predicted ML score (0–1) returned by the server.
    label : Optional[str]
        Classification label returned by the server
        (``"Anti-Cancer Peptide"`` or ``"Non Anti-Cancer Peptide"``).
    raw_html : Optional[str]
        Full HTML of the result page (stored for debugging; not exported to CSV).
    error : Optional[str]
        Error message if the prediction failed, otherwise ``None``.
    """

    peptide_name: str
    peptide: str
    method: str
    threshold: str
    score: Optional[float] = None
    label: Optional[str] = None
    raw_html: Optional[str] = field(default=None, repr=False)
    error: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def success(self) -> bool:
        """True when the prediction completed without errors."""
        return self.error is None

    def to_dict(self, *, include_raw_html: bool = False) -> dict:
        """Return a plain dictionary representation."""
        d = {
            "peptide_name": self.peptide_name,
            "peptide":      self.peptide,
            "method":       self.method,
            "threshold":    self.threshold,
            "score":        self.score,
            "label":        self.label,
            "error":        self.error,
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

        fieldnames = [
            "peptide_name", "peptide", "method", "threshold",
            "score", "label", "error",
        ]
        with path.open("w", newline="", encoding=encoding) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())

        return path.resolve()
