"""
Data models for SEP-AlgPro results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PredictionResult:
    peptide: str
    method: str = "SEP-AlgPro"
    score: Optional[float] = None
    label: Optional[str] = None
    properties: Optional[dict[str, Any]] = None
    raw_html: Optional[str] = field(default=None, repr=False)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None
