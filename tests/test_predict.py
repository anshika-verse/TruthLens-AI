"""
Tests for inference, prediction pipeline, and error handling in TruthLens AI.
"""

import os
import pytest
from predict import TruthLensPredictor, LABEL_MAP
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def test_predictor_missing_model_graceful_handling():
    # Point to a guaranteed non-existent file
    predictor = TruthLensPredictor(model_path="models/non_existent_model.joblib")
    assert predictor.is_model_loaded() is False

    result = predictor.predict("Sample article text that would normally be analyzed.")
    assert result["success"] is False
    assert "not found" in result["error"].lower()
    assert result["label"] is None


def test_predictor_invalid_inputs():
    predictor = TruthLensPredictor()
    # Even if model is not loaded, invalid input validation runs first
    result_empty = predictor.predict("")
    assert result_empty["success"] is False
    assert "empty" in result_empty["error"].lower() or "whitespace" in result_empty["error"].lower()

    result_none = predictor.predict(None)
    assert result_none["success"] is False
    assert "empty" in result_none["error"].lower() or "none" in result_none["error"].lower()


def test_predictor_with_fitted_pipeline(tmp_path):
    # Create and fit a miniature pipeline to test inference and feature explanation
    texts = [
        "breaking scandalous conspiracy secret global plot exposed",
        "shocking leaked documents reveal dark hidden truth",
        "official government department announced bipartisan infrastructure budget",
        "reuters reports that economic statistics show stable quarterly growth"
    ]
    labels = [0, 0, 1, 1]  # 0: Fake, 1: Real

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ])
    pipe.fit(texts, labels)

    # Save to temporary path
    test_model_path = str(tmp_path / "test_model.joblib")
    import joblib
    joblib.dump(pipe, test_model_path)

    test_predictor = TruthLensPredictor(model_path=test_model_path)
    assert test_predictor.is_model_loaded() is True

    # Test prediction on real-sounding text
    res = test_predictor.predict("Government official announced budget growth according to reports.")
    assert res["success"] is True
    assert res["label"] in ["Real", "Fake"]
    assert res["confidence"] is not None
    assert isinstance(res["influential_features"], list)
    assert "TruthLens AI provides statistical classification" in res["disclaimer"]
