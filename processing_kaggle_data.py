import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1) Load data
# Load Kaggle dataset
kaggle_data = pd.read_csv("raw_data/Kaggle_data.csv")

# Same text appearing multiple times
kaggle_duplicate_texts = (
    kaggle_data
    .groupby("text")
    .size()
    .reset_index(name="n_rows")
    .query("n_rows > 1")
    .sort_values("n_rows", ascending=False)
)

print("Number of duplicated text strings:", len(kaggle_duplicate_texts))
print(kaggle_duplicate_texts.head(10))

kaggle_data_no_duplicates = kaggle_data.drop_duplicates(subset="text").copy()

print("Initial number of rows:", len(kaggle_data))
print("Number of rows after removing duplicate texts:", len(kaggle_data_no_duplicates))
print("Number of rows removed:", len(kaggle_data) - len(kaggle_data_no_duplicates))

kaggle_texts_bad_start = kaggle_data_no_duplicates[
    ~kaggle_data_no_duplicates["text"].str.match(r'^\s*[A-Za-z"]', na=False)
]

kaggle_data_clean = kaggle_data_no_duplicates[
    ~kaggle_data_no_duplicates["text"].str.contains(r"\[[^\[\]]+\]", regex=True, na=False)
].copy()

print("Before:", len(kaggle_data_no_duplicates))
print("After:", len(kaggle_data_clean))
print("Removed:", len(kaggle_data_no_duplicates) - len(kaggle_data_clean))

kaggle_label_summary = pd.DataFrame({
    "count": kaggle_data_clean["generated"].value_counts(),
    "proportion": kaggle_data_clean["generated"].value_counts(normalize=True),
    "percentage": kaggle_data_clean["generated"].value_counts(normalize=True) * 100
})

# =========================================================
# 1) Keep only useful columns
# =========================================================
kaggle_data_model = kaggle_data_clean[["text", "generated"]].copy()

# =========================================================
# 2) Split into train / validation / test
#    60% train, 20% validation, 20% test
# =========================================================
kaggle_train, kaggle_temp = train_test_split(
    kaggle_data_model,
    test_size=0.40,
    random_state=42,
    stratify=kaggle_data_model["generated"]
)

kaggle_validation, kaggle_test = train_test_split(
    kaggle_temp,
    test_size=0.50,
    random_state=42,
    stratify=kaggle_temp["generated"]
)

print("Train size:", len(kaggle_train))
print("Validation size:", len(kaggle_validation))
print("Test size:", len(kaggle_test))

#Save the three datasets

output_dir = Path("clean_data/Kaggle")
output_dir.mkdir(parents=True, exist_ok=True)

kaggle_train.to_csv(output_dir / "kaggle_train.csv", index=False)
kaggle_validation.to_csv(output_dir / "kaggle_validation.csv", index=False)
kaggle_test.to_csv(output_dir / "kaggle_test.csv", index=False)