# LEARNING_LOG.md

## What I learned
- Simple bag‑of‑words features (TF‑IDF) combined with linear classifiers can be surprisingly effective for spam detection.  Naive Bayes is fast and leverages conditional independence assumptions【307774325658384†L127-L149】, while logistic regression models the log‑odds of spam and provides calibrated probabilities【66734902495573†L838-L870】.
- Using stratified splits prevents class imbalance from skewing the test set, and fixing a random seed makes experiments reproducible.
- Clear documentation and modular code make it easier to add new models or datasets.

## Mistakes & Fixes
- Initially forgot to normalise the `label` column to lower case, causing mismatches.  Fixed by stripping and lower‑casing labels in the loader.
- At first we split the data without `stratify`, leading to an unbalanced test set.  Fixed by passing `stratify=y` to `train_test_split`.

## Next Improvements
- Experiment with character n‑grams to handle obfuscated spam text.
- Tune the classification threshold to optimise precision (minimise false positives) or recall (catch more spam) depending on the use case.
- Evaluate on a modern email spam corpus to assess generalisation beyond the Enron era.

## Debugging Diary
- Verified that pipelines can be serialised and deserialised using `joblib` without losing performance.
- Ensured that evaluation metrics are computed only on the test split and that the confusion matrix orientation (rows=true, columns=predicted) is consistent.