"""
SEP-AlgPro — Allergenic Protein / Peptide Prediction Scraper
https://balalab-skku.org/SEP-AlgPro/
"""

from .models import PredictionResult
from .predictor import Predictor

__all__ = ["Predictor", "PredictionResult"]
