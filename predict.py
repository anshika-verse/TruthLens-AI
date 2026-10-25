"""
TruthLens AI - Prediction & Explainability Module
Loads the persisted scikit-learn pipeline, validates inputs, predicts binary class,
computes decision score / probability, and extracts influential text features.
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import joblib

from preprocessing import validate_input_text, clean_text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "truthlens_model.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "models", "evaluation_results.json")

LABEL_MAP = {0: "Fake", 1: "Real"}


class TruthLensPredictor:
    """
    Inference and explanation engine for TruthLens AI.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception as e:
                self.pipeline = None
                print(f"[TruthLensPredictor] Error loading model from {self.model_path}: {e}")
        else:
            self.pipeline = None

    def is_model_loaded(self) -> bool:
        return self.pipeline is not None

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Validates text, applies preprocessing, and runs inference.
        Returns a structured dictionary with predictions, confidence score, and explanations.
        """
        # 1. Validation
        is_valid, error_msg = validate_input_text(text)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "prediction": None,
                "label": None,
                "confidence": None,
                "influential_features": [],
                "disclaimer": "No prediction could be made due to invalid input."
            }

        # 2. Check model availability
        if not self.is_model_loaded():
            self._load_model()
            if not self.is_model_loaded():
                return {
                    "success": False,
                    "error": (
                        "Trained model artifact not found. "
                        "Please train the model first using 'python train.py' after placing datasets in data/."
                    ),
                    "prediction": None,
                    "label": None,
                    "confidence": None,
                    "influential_features": [],
                    "disclaimer": "Model is not initialized."
                }

        # 3. Clean text
        cleaned = clean_text(text)

        # 4. Predict
        try:
            pred_class = int(self.pipeline.predict([cleaned])[0])
            label = LABEL_MAP.get(pred_class, "Unknown")

            # Extract components and transform text features
            clf = self.pipeline.named_steps["classifier"]
            tfidf = self.pipeline.named_steps["tfidf"]
            vec = tfidf.transform([cleaned])

            confidence = None
            confidence_type = "Score"

            if hasattr(self.pipeline, "predict_proba"):
                probs = self.pipeline.predict_proba([cleaned])[0]
                confidence = float(probs[pred_class])
                confidence_type = "Class Probability"
            elif hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(vec)[0]
                confidence = float(probs[pred_class])
                confidence_type = "Class Probability"
            elif hasattr(clf, "decision_function"):
                decision = float(clf.decision_function(vec)[0])
                # Sigmoid approximation for LinearSVC
                prob_real = 1.0 / (1.0 + np.exp(-decision))
                confidence = float(prob_real if pred_class == 1 else (1.0 - prob_real))
                confidence_type = "Decision Function (Sigmoid Transformed)"

            # 5. Extract influential features
            influential_features = self._extract_influential_features(cleaned, tfidf, clf, pred_class)

            disclaimer = (
                "IMPORTANT: TruthLens AI provides statistical classification based on patterns "
                "in its training data. This prediction reflects textual similarity and stylistic features, "
                "NOT an authoritative determination of factual truth. Always verify claims through reputable, "
                "independent journalistic sources."
            )

            return {
                "success": True,
                "error": None,
                "prediction": pred_class,
                "label": label,
                "confidence": round(confidence, 4) if confidence is not None else None,
                "confidence_type": confidence_type,
                "cleaned_text": cleaned,
                "influential_features": influential_features,
                "disclaimer": disclaimer
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Inference error: {str(e)}",
                "prediction": None,
                "label": None,
                "confidence": None,
                "influential_features": [],
                "disclaimer": "An error occurred during inference."
            }

    def _extract_influential_features(
        self,
        cleaned_text: str,
        tfidf,
        clf,
        pred_class: int,
        top_n: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Identifies non-zero TF-IDF tokens in the input and ranks them by their
        contribution to the model's decision.
        """
        feature_contributions = []
        try:
            vec = tfidf.transform([cleaned_text])
            feature_names = tfidf.get_feature_names_out()
            nonzero_indices = vec.nonzero()[1]

            if hasattr(clf, "coef_"):
                # Linear models (Logistic Regression, LinearSVC)
                weights = clf.coef_[0]
                for idx in nonzero_indices:
                    token = feature_names[idx]
                    tfidf_val = vec[0, idx]
                    weight = weights[idx]
                    impact = float(tfidf_val * weight)
                    # Positive impact pulls towards Real (1), negative pulls towards Fake (0)
                    favored_label = "Real" if impact > 0 else "Fake"
                    feature_contributions.append({
                        "token": token,
                        "impact": round(abs(impact), 4),
                        "direction": favored_label,
                        "is_aligned_with_prediction": favored_label == LABEL_MAP.get(pred_class)
                    })
            elif hasattr(clf, "feature_log_prob_"):
                # MultinomialNB
                diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
                for idx in nonzero_indices:
                    token = feature_names[idx]
                    impact = float(vec[0, idx] * diff[idx])
                    favored_label = "Real" if impact > 0 else "Fake"
                    feature_contributions.append({
                        "token": token,
                        "impact": round(abs(impact), 4),
                        "direction": favored_label,
                        "is_aligned_with_prediction": favored_label == LABEL_MAP.get(pred_class)
                    })

            # Sort descending by magnitude of impact
            feature_contributions.sort(key=lambda x: x["impact"], reverse=True)
            return feature_contributions[:top_n]
        except Exception:
            return []


def get_evaluation_metrics(metrics_path: str = METRICS_PATH) -> Optional[Dict[str, Any]]:
    """
    Reads saved evaluation metrics if available.
    """
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None
