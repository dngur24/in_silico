"""
AlgPred 2.0 — Allergenic Protein / Peptide Prediction Scraper
https://webs.iiitd.edu.in/raghava/algpred2/batch.html
"""

from .models import Method, PredictionResult
from .predictor import Predictor

__all__ = ["Predictor", "PredictionResult", "Method"]
