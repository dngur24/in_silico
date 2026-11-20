"""
Data models and parameter enums for HemoPI 2.0 web server results.

Target: https://webs.iiitd.edu.in/raghava/hemopi2/prediction.html
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class PredictionMethod(str, Enum):
    """
    Prediction mode options ('Select the method') on the HemoPI 2.0 form.

    - REGRESSION     : Predicts quantitative HC50/EC50 (μM) using RFR based model.
    - CLASSIFICATION : Predicts qualitative Hemolytic / Non-Hemolytic class using
                       ESM2-t6, RF, or MERCI-hybrid models with a score threshold.
    """

    REGRESSION = "regression"
    CLASSIFICATION = "classification"

    @property
    def display_name(self) -> str:
        return {
            "regression": "Regression",
            "classification": "Classification",
        }[self.value]

    @classmethod
    def resolve(cls, value: "PredictionMethod | str | None") -> "PredictionMethod":
        if isinstance(value, cls):
            return value
        if value is None or str(value).strip() == "":
            return cls.REGRESSION
        norm = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        alias_map = {
            "regression": cls.REGRESSION,
            "reg": cls.REGRESSION,
            "r": cls.REGRESSION,
            "classification": cls.CLASSIFICATION,
            "classify": cls.CLASSIFICATION,
            "class": cls.CLASSIFICATION,
            "cls": cls.CLASSIFICATION,
            "c": cls.CLASSIFICATION,
        }
        resolved = alias_map.get(norm)
        if resolved is None:
            raise ValueError(
                f"Invalid HemoPI 2.0 method '{value}'. "
                "Valid options: 'Regression', 'Classification'."
            )
        return resolved


class MLTechnique(str, Enum):
    """
    'Select Machine Learning Technique used for developing model' (form field: `terminus`).

    Regression:
      - '0' : RFR based (default for Regression)

    Classification:
      - '1' : ESM2-t6 based               (default threshold: 0.58)
      - '2' : Hybrid1 (ESM2-t6 + MERCI)   (default for Classification, threshold: 0.58)
      - '3' : RF based                    (default threshold: 0.45)
      - '4' : Hybrid2 (RF + MERCI)        (default threshold: 0.45)
    """

    RFR_BASED = "0"
    ESM2_T6 = "1"
    HYBRID1_ESM2_MERCI = "2"
    RF_BASED = "3"
    HYBRID2_RF_MERCI = "4"

    @property
    def display_name(self) -> str:
        return {
            "0": "RFR based",
            "1": "ESM2-t6 based",
            "2": "Hybrid1 (ESM2-t6 + MERCI)",
            "3": "RF based",
            "4": "Hybrid2 (RF + MERCI)",
        }[self.value]

    @property
    def method(self) -> PredictionMethod:
        if self is MLTechnique.RFR_BASED:
            return PredictionMethod.REGRESSION
        return PredictionMethod.CLASSIFICATION

    @property
    def default_threshold(self) -> Optional["ThresholdValue"]:
        if self is MLTechnique.RFR_BASED:
            return None
        if self in (MLTechnique.ESM2_T6, MLTechnique.HYBRID1_ESM2_MERCI):
            return ThresholdValue.T_0_58
        return ThresholdValue.T_0_45

    @classmethod
    def default_for(cls, method: PredictionMethod) -> "MLTechnique":
        if method is PredictionMethod.REGRESSION:
            return cls.RFR_BASED
        return cls.HYBRID1_ESM2_MERCI

    @classmethod
    def resolve(
        cls,
        value: "MLTechnique | str | None",
        method: PredictionMethod = PredictionMethod.REGRESSION,
    ) -> "MLTechnique":
        if value is None or str(value).strip() == "":
            return cls.default_for(method)

        if isinstance(value, cls):
            resolved = value
        else:
            raw = str(value).strip().lower()
            norm = (
                raw.replace("+", " ")
                .replace("(", " ")
                .replace(")", " ")
                .replace("-", "_")
            )
            norm = "_".join(norm.split())
            alias_map = {
                "0": cls.RFR_BASED,
                "rfr": cls.RFR_BASED,
                "rfr_based": cls.RFR_BASED,
                "random_forest_regressor": cls.RFR_BASED,
                "1": cls.ESM2_T6,
                "esm2": cls.ESM2_T6,
                "esm2_t6": cls.ESM2_T6,
                "esm2_t6_based": cls.ESM2_T6,
                "2": cls.HYBRID1_ESM2_MERCI,
                "hybrid1": cls.HYBRID1_ESM2_MERCI,
                "hybrid_1": cls.HYBRID1_ESM2_MERCI,
                "hybrid1_esm2_t6_merci": cls.HYBRID1_ESM2_MERCI,
                "esm2_t6_merci": cls.HYBRID1_ESM2_MERCI,
                "3": cls.RF_BASED,
                "rf": cls.RF_BASED,
                "rf_based": cls.RF_BASED,
                "random_forest": cls.RF_BASED,
                "4": cls.HYBRID2_RF_MERCI,
                "hybrid2": cls.HYBRID2_RF_MERCI,
                "hybrid_2": cls.HYBRID2_RF_MERCI,
                "hybrid2_rf_merci": cls.HYBRID2_RF_MERCI,
                "rf_merci": cls.HYBRID2_RF_MERCI,
            }
            resolved = alias_map.get(norm)
            if resolved is None:
                raise ValueError(
                    f"Invalid HemoPI 2.0 ML technique '{value}'. "
                    "Valid options: 'RFR based' (Regression); "
                    "'ESM2-t6 based', 'Hybrid1 (ESM2-t6 + MERCI)', "
                    "'RF based', 'Hybrid2 (RF + MERCI)' (Classification)."
                )

        if resolved.method is not method:
            if method is PredictionMethod.REGRESSION:
                raise ValueError(
                    f"Technique '{resolved.display_name}' is for Classification, "
                    "not Regression. For Regression, use 'RFR based'."
                )
            raise ValueError(
                f"Technique '{resolved.display_name}' is for Regression only, "
                "not Classification. For Classification, use one of: "
                "'ESM2-t6 based', 'Hybrid1 (ESM2-t6 + MERCI)', "
                "'RF based', 'Hybrid2 (RF + MERCI)'."
            )
        return resolved


class ThresholdValue(str, Enum):
    """
    Classification threshold options (form field: `th`) on HemoPI 2.0.
    Mirrors the exact `<select name="th">` values on `prediction.html`.
    """

    T_0_0 = "0.0"
    T_0_1 = "0.1"
    T_0_2 = "0.2"
    T_0_3 = "0.3"
    T_0_4 = "0.4"
    T_0_45 = "0.45"
    T_0_5 = "0.5"
    T_0_58 = "0.58"
    T_0_6 = "0.6"
    T_0_7 = "0.7"
    T_0_8 = "0.8"
    T_0_9 = "0.9"
    T_1_0 = "1.0"

    @classmethod
    def resolve(
        cls,
        value: "ThresholdValue | float | str | None",
        default: "ThresholdValue" = T_0_58,
    ) -> "ThresholdValue":
        if isinstance(value, cls):
            return value
        if value is None or str(value).strip() == "":
            return default
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid HemoPI 2.0 threshold '{value}'.") from exc

        for special_num, member in ((0.45, cls.T_0_45), (0.58, cls.T_0_58)):
            if abs(num - special_num) < 1e-6:
                return member

        formatted = f"{round(num, 1):.1f}"
        for member in cls:
            if member.value == formatted:
                return member

        valid = [e.value for e in cls]
        raise ValueError(
            f"Unsupported HemoPI 2.0 threshold '{value}'. Valid options: {valid}"
        )


@dataclass
class PredictionResult:
    """Structured prediction result for one peptide from HemoPI 2.0."""

    peptide: str
    method: str
    score: Optional[float] = None
    label: Optional[str] = None
    properties: Optional[dict[str, Any]] = None
    peptide_name: Optional[str] = None
    technique: Optional[str] = None
    threshold: Optional[str] = None
    job_id: Optional[str] = None
    visual_presentation: Optional[list[dict[str, Any]]] = None
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
            "technique": self.technique,
            "threshold": self.threshold,
            "score": self.score,
            "label": self.label,
            "properties": self.properties,
            "visual_presentation": self.visual_presentation,
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
            "technique",
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
