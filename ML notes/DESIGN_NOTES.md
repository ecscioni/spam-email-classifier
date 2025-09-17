# DESIGN_NOTES.md

## Assumptions
- Labels: `ham` (legitimate) and `spam`; email text is primarily in English and includes both subject and body.  We treat the subject and body concatenated into a single `text` field.
- Classical machine learning models (TF‑IDF with Naive Bayes or Logistic Regression) are sufficient for a one‑week scope and provide interpretable baselines.
- The Enron corpus (early‑2000s corporate emails) is used for demonstration; we acknowledge that language and tactics in modern spam have shifted.

## Architecture

```
Raw CSV → split (train/test)
     |→ Pipeline A: TF‑IDF → MultinomialNB → metrics
     |→ Pipeline B: TF‑IDF → LogisticRegression → metrics
Save best → predict_cli.py
```

## Key Decisions
- **TF‑IDF features** convert text to numerical vectors while down‑weighting common words.  This is a standard and strong baseline for text classification.
- **Multinomial Naive Bayes** applies Bayes’ theorem with a “naive” independence assumption and is particularly effective and fast on text classification【307774325658384†L127-L149】.
- **Logistic Regression** is implemented in scikit‑learn as a linear classifier that models class probabilities using a logistic (sigmoid) function. It supports binary and multinomial tasks and incorporates regularisation【66734902495573†L838-L870】.
- **Hyperparameter sweep**: we search over regularisation strength `C` values `[0.5, 1.0, 2.0]` for logistic regression to pick the best F1.

## Trade‑offs & Alternatives
- **Character n‑grams** could capture obfuscation and misspellings but dramatically increase feature dimensionality.
- **HashingVectorizer** offers constant memory and streaming support at the cost of interpretability.
- **Modern embeddings (e.g. BERT/SBERT)** capture semantics better but require heavy computation and may be overkill for a baseline.

## Testing Strategy
- **Unit tests** ensure the CSV loader returns the expected schema and the pipelines train and predict without errors on the sample data.
- **Manual checks** call the CLI with obviously spammy and hammy messages to verify outputs.
- **Reproducibility**: fix a random seed (e.g. 42) and use a deterministic split.

## References
- Naive Bayes assumptions and effectiveness for text classification【307774325658384†L127-L149】.
- Logistic regression as a linear classifier using a logistic function and regularisation【66734902495573†L838-L870】.
- Enron dataset description and size【14671641396911†L54-L117】.