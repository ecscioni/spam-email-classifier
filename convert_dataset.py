import pandas as pd

# Load the raw Enron dataset
df = pd.read_csv("data/enron_spam.csv")

# Combine Subject + Message into one text field
df["text"] = df["Subject"].fillna("") + " " + df["Message"].fillna("")

# Rename Spam/Ham column to 'label' and lowercase
df = df.rename(columns={"Spam/Ham": "label"})
df["label"] = df["label"].str.lower()

# Keep only the two columns we need
df = df[["label", "text"]]

# Save clean version
df.to_csv("data/enron_clean.csv", index=False)
print("✅ Wrote data/enron_clean.csv with shape:", df.shape)
print(df.head(3))
