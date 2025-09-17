# PROJECT_PLAN.md

## Goal
Train two classic ML models (Naive Bayes and Logistic Regression) to classify email messages as spam or ham and critically analyse performance & risks within one week.

## Scope (week‑sized)
- Implement pipelines using TF‑IDF features.
- Evaluate accuracy, precision, recall and F1; produce a confusion matrix.
- Provide a command line interface (CLI) for ad‑hoc predictions.
- Reflect on which metric matters most and discuss adversarial tactics.
- Document everything comprehensively.

## Requirements Table
| Category | Items |
| --- | --- |
| Functional | Train NB and LR models; save the best model; CLI predict |
| Data | CSV with columns: `label`, `text` (ham/spam) |
| Evaluation | Accuracy, Precision, Recall, F1; confusion matrix |
| Deliverables | Code, documentation, metrics, CLI, tests |
| Integrity | Cite datasets & libraries; maintain `AI_USAGE.md` |

## Milestones & Timeline
- **Day 1**: Repository skeleton, virtual environment, sample data, loader, and test scaffolding.
- **Day 2**: TF‑IDF + Multinomial Naive Bayes pipeline and run baseline.
- **Day 3**: TF‑IDF + Logistic Regression with a small hyperparameter sweep.
- **Day 4**: Validation, confusion matrix plot, and error analysis.
- **Day 5**: Documentation, packaging, rubric self‑check and final polish.

## Risks & Mitigations
- **Dataset bias/domain shift** – The Enron corpus consists of early‑2000s corporate emails; its vocabulary and spam tactics differ from today. Document limitations and propose cross‑domain evaluation on more recent datasets.
- **Adversarial text** – Spammers may obfuscate with misspellings, HTML and images; discuss char n‑grams and continual retraining as mitigations.
- **Reproducibility** – Use fixed random seeds, pin dependency versions, and record the environment to ensure results can be replicated.

## Success Criteria
- Reproducible training pipelines.
- Evaluation metrics produced and saved.
- Clear reflection on metric trade‑offs.
- All tests pass and documentation is complete.