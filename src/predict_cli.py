"""
Command‑line interface for classifying a single email as spam or ham.

Usage:

```bash
python src/predict_cli.py "Subject: Free offer! Click here now."
```

The script loads the saved best model (models/best_model.joblib) and prints
the predicted label and probability.
"""

from __future__ import annotations

import argparse
import joblib


def main(text: str) -> None:
    """Load the saved model and classify the given text.

    Parameters
    ----------
    text : str
        The email text to classify (subject and body combined).
    """
    model = joblib.load("models/best_model.joblib")
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    # The positive class corresponds to index 1
    spam_prob = float(proba[1])
    label = "spam" if pred == 1 else "ham"
    print(f"label={label}  prob={spam_prob:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict whether an email is spam or ham.")
    parser.add_argument("text", type=str, help="The email text (wrap in quotes)")
    args = parser.parse_args()
    main(args.text)