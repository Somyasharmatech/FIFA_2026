"""SHAP-based explainability and prediction narratives."""

from src.explainability.explainer import ModelExplainer
from src.explainability.narratives import generate_match_narrative

__all__ = ["ModelExplainer", "generate_match_narrative"]
