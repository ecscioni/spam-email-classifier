PY=python

train:
	$(PY) src/train.py --data data/enron_sample.csv --out_dir .

eval:
	$(PY) src/evaluate.py --data data/enron_sample.csv --model models/best_model.joblib --out_dir reports

predict:
	$(PY) src/predict_cli.py "Subject: This is a test email."

test:
	pytest -q