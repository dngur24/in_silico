"""
ToxinPred 3.0 Web Scraper Package

Prediction of Toxic and Non-Toxic Peptides
Target: https://webs.iiitd.edu.in/raghava/toxinpred3/prediction.php
"""

from .models import (
    MLTechnique,
    PredictionResult,
    ThresholdValue,
)
from .parser import Parser
from .predictor import Predictor

# Convenient alias consistent with other in_silico packages
Method = MLTechnique

__all__ = [
    "Predictor",
    "Parser",
    "PredictionResult",
    "MLTechnique",
    "Method",
    "ThresholdValue",
]
__version__ = "1.0.0"
