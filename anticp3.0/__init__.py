"""
AntiCP 3.0 — Anticancer Protein / Peptide Prediction Scraper
https://webs.iiitd.edu.in/raghava/anticp3/predict.html
"""

from .predictor import Method, PredictionResult, Predictor

__all__ = ["Predictor", "PredictionResult", "Method"]
