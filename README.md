# README.md
# 📧 Spam Email Classification — Week Project

**Goal**: Build and compare two classic text classifiers (Multinomial Naive Bayes and Logistic Regression) to detect spam vs. ham in **emails**, report evaluation metrics, and reflect on the best operating metric.

## Why this matters
Email users face an ongoing flood of unwanted messages. Effective spam filtering protects users from phishing, fraud and wasted time. This project teaches how to turn raw email text into features, train interpretable models and evaluate them.

## Dataset
Primary dataset: **Enron Spam Email dataset**. A version of the Enron email collection labelled as `ham` (legitimate) or `spam`, with subject and message merged into a text field. The dataset contains about 33.7k rows with a train/test split of 31.7k/2k messages【14671641396911†L54-L62】. Each record includes a `message_id`, the `text` (subject+message), a binary `label`, a `label_text` string, plus `subject`, `message` and `date` columns【14671641396911†L54-L117】.  You can download the full dataset from Hugging Face or Kaggle; see `DATA_CARD.md` for details.

For portability, this repository contains a **small sample file** at `data/enron_sample.csv` with five examples to let you run the code out‑of‑the‑box. Replace it with the full dataset for training a real model.

## Setup (cross‑platform)

```bash
# Windows PowerShell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# 1) Train both models; saves the best model to models/best_model.joblib
python src/train.py --data data/enron_clean.csv --out_dir .

# 2) Evaluate on test split; writes metrics.json & confusion_matrix.png in reports/
python src/evaluate.py --data data/enron_clean.csv --model models/best_model.joblib --out_dir reports

# 3) Predict via CLI
python src/predict_cli.py \"Subject: Congrats! You won a free prize! Claim now.\"
python src/predict_cli.py \"Subject: Meeting agenda attached. See you tomorrow.\"
```

Example output:

```
Model: tfidf+NB      accuracy=0.97 precision=0.96 recall=0.93 f1=0.94
Model: tfidf+LogReg  accuracy=0.98 precision=0.97 recall=0.95 f1=0.96
Saved best model to models/best_model.joblib
```

Your scores will vary with the data split and dataset size.

## How it works (short)
- **Vectorization**: `TfidfVectorizer` converts emails into a bag‑of‑words representation.  This highlights important terms by weighting rare words more heavily than common words.
- **Models**: 
  - **Multinomial Naive Bayes** assumes conditional independence of features and is particularly fast and effective on text【307774325658384†L127-L149】.
  - **Logistic Regression** models the probability of spam using a logistic (sigmoid) function. In scikit‑learn it is implemented as a linear classifier with regularisation; probabilities are converted to classes via a threshold【66734902495573†L838-L870】.
- **Metrics**: we report accuracy, precision, recall and F1 score; see `TESTING.md` for details.

## Licences & Data Use
Datasets remain under their original licences. Always cite the source if you publish results. See `DATA_CARD.md` for details.