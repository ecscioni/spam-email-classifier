"""
Basic tests for the spam email classification project.

These tests ensure that the data loader works and that the training pipeline can
fit and predict on the sample data without errors.  They are intentionally
lightweight to run quickly.
"""

from __future__ import annotations

from src.prepare_data import load_dataset  # type: ignore
from src.train import build_nb_pipeline  # type: ignore


def test_load_schema() -> None:
    df = load_dataset("data/enron_sample.csv")
    assert {"label", "text"}.issubset(set(df.columns))
    assert len(df) >= 3


def test_train_pipeline() -> None:
    df = load_dataset("data/enron_sample.csv")
    X = df["text"].tolist()
    y = (df["label"] == "spam").astype(int).tolist()
    pipe = build_nb_pipeline()
    pipe.fit(X, y)
    p = pipe.predict([X[0]])
    assert p.shape == (1,)