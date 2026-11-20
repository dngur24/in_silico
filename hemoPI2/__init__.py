"""
HemoPI 2.0 Web Scraper Package

Prediction of Hemolytic Activity / Potency (HC50 and Classification) of Peptides
Target: https://webs.iiitd.edu.in/raghava/hemopi2/prediction.html
"""

from .models import (
    MLTechnique,
    PredictionMethod,
    PredictionResult,
    ThresholdValue,
)
from .parser import Parser
from .predictor import Predictor

# Convenient alias consistent with other in_silico packages
Method = PredictionMethod

__all__ = [
    "Predictor",
    "Parser",
    "PredictionResult",
    "PredictionMethod",
    "Method",
    "MLTechnique",
    "ThresholdValue",
]
__version__ = "1.0.0"
