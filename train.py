"""
TruthLens AI - Training & Model Evaluation Pipeline
Loads datasets, performs validation, fits candidate ML pipelines (Logistic Regression,
Multinomial Naive Bayes, Linear SVM), compares evaluation metrics, and persists the best model.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TruthLens-Train")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "truthlens_model.joblib")
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")

# Explicit binary label mapping
LABEL_MAP = {
    0: "Fake",
    1: "Real"
}


def load_and_validate_dataset(data_dir: str = DATA_DIR) -> Optional[pd.DataFrame]:
    """
    Looks for data/Fake.csv and data/True.csv (standard Kaggle Fake and Real News dataset format)
    or a single combined dataset in data/.
    Validates columns, missing rows, duplicates, and class balance.
    """
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    if os.path.exists(fake_path) and os.path.exists(true_path):
        logger.info(f"Found dataset files: {fake_path} and {true_path}")
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)

        # Label assignment: 0 = Fake, 1 = Real
        df_fake["label"] = 0
        df_true["label"] = 1

        # Check required text columns
        for name, df in [("Fake.csv", df_fake), ("True.csv", df_true)]:
            if "title" not in df.columns and "text" not in df.columns:
                raise ValueError(f"Dataset {name} missing both 'title' and 'text' columns.")

        df_combined = pd.concat([df_fake, df_true], ignore_index=True)
    else:
        # Check if a single combined CSV exists in data/
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            if csv_files:
                combined_path = os.path.join(data_dir, csv_files[0])
                logger.info(f"Loading single dataset file: {combined_path}")
                df_combined = pd.read_csv(combined_path)
            else:
                logger.warning(f"No CSV files found in '{data_dir}'.")
                return None
        else:
            logger.warning(f"Data directory '{data_dir}' does not exist.")
            return None

    # Construct unified text column: combine title + text where available
    if "title" in df_combined.columns and "text" in df_combined.columns:
        df_combined["content"] = df_combined["title"].fillna("") + " " + df_combined["text"].fillna("")
    elif "text" in df_combined.columns:
        df_combined["content"] = df_combined["text"].fillna("")
    elif "title" in df_combined.columns:
        df_combined["content"] = df_combined["title"].fillna("")
    else:
        raise ValueError("Dataset does not contain 'text' or 'title' columns.")

    # Remove rows where content is empty or whitespace
    df_combined["content"] = df_combined["content"].astype(str).str.strip()
    df_clean = df_combined[df_combined["content"].str.len() > 10].copy()

    # Drop exact content duplicates to avoid train/test leakage
    init_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["content"]).copy()
    logger.info(f"Deduplication: removed {init_len - len(df_clean)} duplicate rows. Total unique articles: {len(df_clean)}.")

    # Validate class distribution
    class_counts = df_clean["label"].value_counts().to_dict()
    logger.info(f"Class distribution: {class_counts} (0=Fake: {class_counts.get(0, 0)}, 1=Real: {class_counts.get(1, 0)})")

    return df_clean


def evaluate_pipeline(
    pipeline: Pipeline,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    target_names: list = ["Fake", "Real"]
) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics: Accuracy, Precision, Recall, F1, and Confusion Matrix.
    """
    y_pred = pipeline.predict(X_eval)

    acc = float(accuracy_score(y_eval, y_pred))
    prec = float(precision_score(y_eval, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_eval, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_eval, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_eval, y_pred).tolist()
    report = classification_report(y_eval, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report
    }


def train_and_compare(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Splits data into train (70%), validation (15%), and test (15%).
    Trains Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
    Selects best model via validation F1-score, then computes final metrics on test set.
    """
    logger.info("Cleaning texts using standardized preprocessing...")
    # Preprocess text
    X_raw = df["content"].apply(clean_text).values
    y = df["label"].values

    # Stratified Train (70%) / Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.30, random_state=42, stratify=y
    )
    # Stratified Val (15%) / Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logger.info(f"Split sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

    # Candidate models
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Linear SVM": LinearSVC(random_state=42, max_iter=2000, C=1.0)
    }

    results = {}
    best_name = None
    best_val_f1 = -1.0
    best_pipeline = None

    for name, clf in candidates.items():
        logger.info(f"Training pipeline for: {name}...")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
            ("classifier", clf)
        ])

        # Fit ONLY on training data to avoid leakage
        pipeline.fit(X_train, y_train)

        # Evaluate on validation data
        val_metrics = evaluate_pipeline(pipeline, X_val, y_val)
        logger.info(f"{name} Validation F1: {val_metrics['f1_score']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

        results[name] = {
            "validation_metrics": val_metrics
        }

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_name = name
            best_pipeline = pipeline

    logger.info(f"Best model based on Validation F1: {best_name} (F1: {best_val_f1:.4f})")

    # Evaluate final selected model on held-out Test set
    test_metrics = evaluate_pipeline(best_pipeline, X_test, y_test)
    logger.info(f"Final Test Evaluation for {best_name}: Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")

    summary = {
        "best_model": best_name,
        "selected_test_metrics": test_metrics,
        "all_models_validation": {k: v["validation_metrics"] for k, v in results.items()},
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "class_distribution": {"Fake (0)": int((y == 0).sum()), "Real (1)": int((y == 1).sum())}
        },
        "label_mapping": LABEL_MAP
    }

    return best_pipeline, summary


def run_training_pipeline() -> Optional[Pipeline]:
    """
    Main entry point for training.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_and_validate_dataset(DATA_DIR)
    if df is None:
        logger.error(
            "\n" + "=" * 60 + "\n"
            "DATASET SETUP REQUIRED:\n"
            "To train TruthLens AI, please place the Kaggle Fake and Real News dataset in the 'data/' directory:\n"
            f"  -> {os.path.join(DATA_DIR, 'Fake.csv')}\n"
            f"  -> {os.path.join(DATA_DIR, 'True.csv')}\n"
            "Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            "Once files are placed, re-run this script: python train.py\n" +
            "=" * 60
        )
        return None

    pipeline, summary = train_and_compare(df)

    # Save pipeline and metrics
    logger.info(f"Saving trained pipeline to {MODEL_SAVE_PATH}...")
    joblib.dump(pipeline, MODEL_SAVE_PATH)

    logger.info(f"Saving evaluation metrics to {METRICS_SAVE_PATH}...")
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training and evaluation completed successfully.")
    return pipeline


if __name__ == "__main__":
    run_training_pipeline()
