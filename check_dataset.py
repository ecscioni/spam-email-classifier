import pandas as pd

df = pd.read_csv("data/enron_spam.csv")

print("Columns found:", df.columns.tolist())
print(df.head(3))
