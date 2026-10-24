"""
Data models for AlgPred 2.0 results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Method(str, Enum):
    """Prediction models available on AlgPred 2.0."""

    AAC_RF = "0"   # AAC based Random Forest
    HYBRID = "4"   # Hybrid (RF + BLAST + MERCI) — default recommended

    @classmethod
    def resolve(cls, value: Optional[str]) -> "Method":
        if isinstance(value, cls):
            return value
        if not value:
            return cls.HYBRID
        norm = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if norm in ("0", "aac", "aac_rf", "rf"):
            return cls.AAC_RF
        if norm in ("4", "hybrid"):
            return cls.HYBRID
        raise ValueError(f"Unknown AlgPred 2.0 method '{value}'. Valid: 'Hybrid', 'AAC_RF'.")


@dataclass
class PredictionResult:
    peptide: str
    method: str
    score: Optional[float] = None
    label: Optional[str] = None
    properties: Optional[dict[str, Any]] = None
    raw_html: Optional[str] = field(default=None, repr=False)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None
