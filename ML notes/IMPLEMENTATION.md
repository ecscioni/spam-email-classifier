# IMPLEMENTATION.md

This document records how the project was built step by step and ties explanations to code sections.

## 1) Data schema & loader

- The expected input is a CSV file with columns `label` and `text`.  The `label` field should contain `ham` for legitimate emails and `spam` for unwanted messages.  The `text` field contains the subject and message combined into a single string.
- We use `pandas.read_csv` to load the data.  Columns are normalised to lower case and leading/trailing whitespace is trimmed.  The loader asserts that the required columns are present.

## 2) Pipelines

We implement two Scikit‑Learn pipelines to encapsulate feature extraction and model training:

1. **Naive Bayes pipeline**: `TfidfVectorizer(ngram_range=(1,2), min_df=2)` followed by `MultinomialNB`.  The TF‑IDF vectoriser converts text into weighted term frequency vectors; the Naive Bayes classifier assumes conditional independence and works well on sparse count data.
2. **Logistic Regression pipeline**: `TfidfVectorizer(ngram_range=(1,2), min_df=2)` followed by `LogisticRegression(C=C, max_iter=200, solver="liblinear")`.  Logistic regression models the log‑odds of spam and supports regularisation.

Both pipelines are built using Scikit‑Learn’s `Pipeline` class, which ensures that the same preprocessing is applied at train and prediction time.

## 3) Training & model selection

- The dataset is split into training and test sets using `train_test_split` with an 80/20 ratio, stratifying by the label and fixing `random_state=42` for reproducibility.
- We train each pipeline on the training set and evaluate on the test set using the F1 score (treating spam as the positive class).  The F1 score is chosen as it balances precision and recall.
- We perform a small hyperparameter sweep over `C ∈ {0.5, 1.0, 2.0}` for the logistic regression model to explore different regularisation strengths.  Each candidate’s F1 is stored.
- The model with the highest F1 is saved to `models/best_model.joblib` using `joblib.dump`.  A JSON summary of candidate models is written to `reports/train_summary.json`.

## 4) Evaluation & reporting

- The evaluation script loads the dataset and the saved model, recomputes the train/test split (using the same seed and stratification), and predicts on the test set.
- It calculates accuracy, precision, recall and F1 using `sklearn.metrics`.  A JSON file `reports/metrics.json` stores the numeric metrics.
- A confusion matrix is generated with `confusion_matrix` and plotted using `matplotlib`.  The resulting PNG is saved to `reports/confusion_matrix.png`.

## 5) CLI predictions

- The CLI script loads the best model from `models/best_model.joblib` and accepts a single email message from the command line.  It outputs the predicted label (`spam` or `ham`) and the predicted probability for the chosen class.

## 6) Commands

To reproduce the pipeline, run the following commands:

```bash
python src/train.py --data data/enron_clean.csv --out_dir .
python src/evaluate.py --data data/enron_clean.csv --model models/best_model.joblib --out_dir reports
python src/predict_cli.py "Subject: Free vacation offer! Click here to claim."
```

## Notes

- `TfidfVectorizer` uses both unigrams and bigrams to capture common two‑word phrases.  The `min_df` parameter filters out words that appear in fewer than two emails to reduce noise.
- The logistic regression solver is set to `liblinear` as it works well on small to medium‑sized datasets.  Increasing `max_iter` ensures convergence.