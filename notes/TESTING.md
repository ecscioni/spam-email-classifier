# TESTING.md

## Test Plan

### Unit tests
- **Schema test**: Load the sample CSV and verify that it contains the `label` and `text` columns and at least a few rows.
- **Training test**: Fit the Naive Bayes pipeline on the sample dataset and ensure that the `predict` method returns an array of the correct shape.

### Manual tests
- **Spam vs ham**: Run the CLI with an obviously spammy email (e.g. “Subject: You won a free iPhone! Click now”) and expect the model to classify it as `spam`.  Then run it with a neutral email (e.g. “Subject: Reminder for tomorrow’s meeting”) and expect `ham`.
- **Confusion matrix**: After evaluation, open `reports/confusion_matrix.png` and confirm it shows a 2×2 matrix.

## Commands

```bash
pytest -q
python src/train.py --data data/enron_clean.csv --out_dir .
python src/evaluate.py --data data/enron_clean.csv --model models/best_model.joblib --out_dir reports
python src/predict_cli.py "Subject: Free vacation offer! Click here to claim."
python src/predict_cli.py "Subject: Meeting notes attached for review."
```

## Expected vs Actual (example)

| Case | Expected | Actual (example) |
| --- | --- | --- |
| Obvious spam (prize offer, free, click) | spam | spam (probability ≈ 0.95) |
| Neutral business email | ham | ham (probability ≈ 0.90) |

## Edge Cases
- Very short messages like “OK” or “Yes” may result in lower confidence; this is acceptable.
- Messages containing both spammy and legitimate content (e.g. forwarded chain letters) may produce ambiguous probabilities.  Threshold tuning is beyond this week’s scope.