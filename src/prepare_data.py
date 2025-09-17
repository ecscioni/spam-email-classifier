"""
Data loading utilities for the spam email classification project.

This module exposes a single function, `load_dataset`, which reads a CSV file
containing a `label` column (ham/spam) and a `text` column (email subject +
body).  It normalises column names, strips whitespace and lower‑cases the
labels.  Any comment lines (starting with `#`) are ignored.
"""

from __future__ import annotations

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Load a labelled email dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.  The file must contain `label` and `text`
        columns.  Comments beginning with `#` are skipped.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with normalised `label` and `text` columns.
    """
    df = pd.read_csv(path, comment="#")
    # Normalise column names
    df = df.rename(columns=lambda c: c.strip().lower())
    # Verify required columns exist
    if not {"label", "text"}.issubset(set(df.columns)):
        raise ValueError("CSV must contain 'label' and 'text' columns")
    # Clean values
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)
    return df