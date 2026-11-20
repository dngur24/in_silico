"""
Data models and parameter enums for ToxinPred 3.0 web server results.

Target: https://webs.iiitd.edu.in/raghava/toxinpred3/prediction.php
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class MLTechnique(str, Enum):
    """
    'Select Machine Learning Technique used for developing model' (form field: `terminus`).

    - '0' : ET based          (default threshold: 0.5)
    - '1' : Hybrid(ET+MERCI)  (default model, default threshold: 0.38)
    - '2' : DL based          (default threshold: 0.5)
    - '3' : Hybrid(DL+MERCI)  (default threshold: 0.38)
    """

    ET_BASED = "0"
    HYBRID_ET_MERCI = "1"
    DL_BASED = "2"
    HYBRID_DL_MERCI = "3"

    @property
    def display_name(self) -> str:
        return {
            "0": "ET based",
            "1": "Hybrid(ET+MERCI)",
            "2": "DL based",
            "3": "Hybrid(DL+MERCI)",
        }[self.value]

    @property
    def default_threshold(self) -> "ThresholdValue":
        if self in (MLTechnique.HYBRID_ET_MERCI, MLTechnique.HYBRID_DL_MERCI):
            return ThresholdValue.T_0_38
        return ThresholdValue.T_0_5

    @classmethod
    def resolve(cls, value: "MLTechnique | str | None") -> "MLTechnique":
        if isinstance(value, cls):
            return value
        if value is None or str(value).strip() == "":
            return cls.HYBRID_ET_MERCI

        raw = str(value).strip().lower()
        norm = (
            raw.replace("+", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace("-", "_")
        )
        norm = "_".join(norm.split())
        alias_map = {
            "0": cls.ET_BASED,
            "et": cls.ET_BASED,
            "et_based": cls.ET_BASED,
            "extra_tree": cls.ET_BASED,
            "extra_trees": cls.ET_BASED,
            "1": cls.HYBRID_ET_MERCI,
            "hybrid_et_merci": cls.HYBRID_ET_MERCI,
            "hybird_et_merci": cls.HYBRID_ET_MERCI,
            "et_merci": cls.HYBRID_ET_MERCI,
            "hybrid_et": cls.HYBRID_ET_MERCI,
            "hybrid": cls.HYBRID_ET_MERCI,
            "2": cls.DL_BASED,
            "dl": cls.DL_BASED,
            "dl_based": cls.DL_BASED,
            "deep_learning": cls.DL_BASED,
            "3": cls.HYBRID_DL_MERCI,
            "hybrid_dl_merci": cls.HYBRID_DL_MERCI,
            "hybird_dl_merci": cls.HYBRID_DL_MERCI,
            "dl_merci": cls.HYBRID_DL_MERCI,
            "hybrid_dl": cls.HYBRID_DL_MERCI,
        }
        resolved = alias_map.get(norm)
        if resolved is None:
            raise ValueError(
                f"Invalid ToxinPred 3.0 technique '{value}'. "
                "Valid options: 'ET based', 'Hybrid(ET+MERCI)', 'DL based', 'Hybrid(DL+MERCI)'."
            )
        return resolved


class ThresholdValue(str, Enum):
    """
    Threshold options (form field: `th`) on ToxinPred 3.0 (`prediction.php`).
    0.0 to 1.0 in 0.1 increments plus 0.38 between 0.3 and 0.4.
    """

    T_0_0 = "0.0"
    T_0_1 = "0.1"
    T_0_2 = "0.2"
    T_0_3 = "0.3"
    T_0_38 = "0.38"
    T_0_4 = "0.4"
    T_0_5 = "0.5"
    T_0_6 = "0.6"
    T_0_7 = "0.7"
    T_0_8 = "0.8"
    T_0_9 = "0.9"
    T_1_0 = "1.0"

    @classmethod
    def resolve(
        cls,
        value: "ThresholdValue | float | str | None",
        default: "ThresholdValue" = T_0_38,
    ) -> "ThresholdValue":
        if isinstance(value, cls):
            return value
        if value is None or str(value).strip() == "":
            return default
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ToxinPred 3.0 threshold '{value}'.") from exc

        if abs(num - 0.38) < 1e-6:
            return cls.T_0_38

        formatted = f"{round(num, 1):.1f}"
        for member in cls:
            if member.value == formatted:
                return member

        valid = [e.value for e in cls]
        raise ValueError(
            f"Unsupported ToxinPred 3.0 threshold '{value}'. Valid options: {valid}"
        )


@dataclass
class PredictionResult:
    """Structured prediction result for one peptide from ToxinPred 3.0."""

    peptide: str
    method: str
    score: Optional[float] = None
    label: Optional[str] = None
    properties: Optional[dict[str, Any]] = None
    peptide_name: Optional[str] = None
    threshold: Optional[str] = None
    job_id: Optional[str] = None
    raw_html: Optional[str] = field(default=None, repr=False)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "peptide_name": self.peptide_name,
            "peptide": self.peptide,
            "method": self.method,
            "threshold": self.threshold,
            "score": self.score,
            "label": self.label,
            "properties": self.properties,
            "job_id": self.job_id,
            "success": self.success,
            "error": self.error,
        }

    @staticmethod
    def to_csv(results: list["PredictionResult"], path: str | Path) -> Path:
        out = Path(path)
        fieldnames = [
            "peptide_name",
            "peptide",
            "method",
            "threshold",
            "score",
            "label",
            "properties",
            "job_id",
            "success",
            "error",
        ]
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                row = res.to_dict()
                if row["properties"] is not None:
                    row["properties"] = json.dumps(row["properties"], ensure_ascii=False)
                writer.writerow(row)
        return out

    @staticmethod
    def to_json(results: list["PredictionResult"], path: str | Path) -> Path:
        out = Path(path)
        payload = [r.to_dict() for r in results]
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return out
