"""
Evaluation script for the spam email classification project.

This script loads a saved model and the dataset, re‑creates the train/test split
with the same seed, evaluates the model on the test set and writes metrics and
a confusion matrix image to the specified output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

from prepare_data import load_dataset


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    """Save a simple confusion matrix plot to a PNG file.

    Parameters
    ----------
    cm : ndarray
        A 2×2 confusion matrix where rows correspond to true labels and columns to predicted labels.
    out_path : pathlib.Path
        File path to write the PNG image.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion matrix (rows=true, cols=pred)")
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(["ham", "spam"])
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(["ham", "spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)


def main(data_path: str, model_path: str, out_dir: str) -> None:
    """Evaluate a saved model and write metrics and a confusion matrix.

    Parameters
    ----------
    data_path : str
        Path to the CSV dataset used for training/testing.
    model_path : str
        Path to the saved joblib model.
    out_dir : str
        Directory where the metrics JSON and confusion matrix image will be saved.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    X = df["text"].values
    y = (df["label"] == "spam").astype(int).values

    # Recreate the train/test split with the same random seed.  For small
    # datasets it may not be possible to stratify (each class must appear at
    # least once in the test set), so we replicate the fallback logic used in
    # the training script.
    test_fraction = 0.2
    test_size_count = int(len(y) * test_fraction)
    class_counts = np.bincount(y)
    stratify_y = None
    if len(class_counts) >= 2 and min(class_counts) >= 2 and test_size_count >= len(class_counts):
        stratify_y = y
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=42, stratify=stratify_y
    )

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Write metrics JSON
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plot_confusion_matrix(cm, out / "confusion_matrix.png")

    # Print classification report to stdout for convenience
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved spam classification model.")
    parser.add_argument("--data", required=True, help="Path to the CSV dataset")
    parser.add_argument("--model", required=True, help="Path to the saved model (joblib)")
    parser.add_argument("--out_dir", default="reports", help="Output directory for metrics and confusion matrix")
    args = parser.parse_args()
    main(args.data, args.model, args.out_dir)