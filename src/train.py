"""
Training script for the spam email classification project.

This script loads a labelled email dataset, builds two pipelines (TF‑IDF +
Multinomial Naive Bayes and TF‑IDF + Logistic Regression), evaluates them
using F1 score on a held‑out test set, selects the best model, and saves
it to disk.  It also writes a JSON summary of all candidate F1 scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from prepare_data import load_dataset


def build_nb_pipeline() -> Pipeline:
    """Return a pipeline of TF‑IDF vectoriser and Multinomial Naive Bayes."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", MultinomialNB()),
    ])


def build_lr_pipeline(C: float = 1.0) -> Pipeline:
    """Return a pipeline of TF‑IDF vectoriser and Logistic Regression.

    Parameters
    ----------
    C : float, optional
        Inverse regularisation strength. Smaller values specify stronger
        regularisation.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(C=C, max_iter=200, solver="liblinear")),
    ])



def main(data_path: str, out_dir: str, limit: int | None = None) -> None:
    """Train models and save the best one.

    Parameters
    ----------
    data_path : str
        Path to the input CSV file.
    out_dir : str
        Directory to store models and report files.  Subdirectories `models/`
        and `reports/` will be created if they do not exist.
    """
    out = Path(out_dir)
    models_dir = out / "models"
    reports_dir = out / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)

    if limit is not None:
        df = df.sample(n=limit, random_state=42)
        print(f"⚡ Using a limited subset: {len(df)} rows")
    
    X = df["text"].values
    # Convert labels: 1 for spam, 0 for ham
    y = (df["label"] == "spam").astype(int).values

    # Determine whether stratified splitting is possible.  A stratified split requires
    # at least one example of each class in the test set.  If the dataset is
    # extremely small (e.g. our sample of five emails), this may not be possible.
    test_fraction = 0.2
    # Number of examples in the test split
    test_size_count = int(len(y) * test_fraction)
    # Count the number of examples per class
    class_counts = np.bincount(y)
    # Use stratification only if each class has at least two instances and the
    # test split will contain at least as many examples as there are classes.
    stratify_y = None
    if len(class_counts) >= 2 and min(class_counts) >= 2 and test_size_count >= len(class_counts):
        stratify_y = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=42, stratify=stratify_y
    )

    candidates: list[tuple[str, Pipeline, float]] = []

    # Train Naive Bayes
    nb_model = build_nb_pipeline()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    candidates.append(("tfidf+NB", nb_model, f1))

    # Train logistic regression with a small C sweep
    for C in [0.5, 1.0, 2.0]:
        lr_model = build_lr_pipeline(C=C)
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        f1_lr = f1_score(y_test, y_pred_lr)
        candidates.append((f"tfidf+LogReg(C={C})", lr_model, f1_lr))

    # Select best model
    best_name, best_model, best_f1 = max(candidates, key=lambda t: t[2])
    joblib.dump(best_model, models_dir / "best_model.joblib")

    # Write summary JSON
    summary = {
        "candidates": [
            {"name": name, "f1": f1_score_value} for name, _, f1_score_value in candidates
        ],
        "best_model": best_name,
        "best_f1": best_f1,
    }
    with open(reports_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: limit number of rows for faster training")
    args = parser.parse_args()
    main(args.data, args.out_dir, limit=args.limit)

