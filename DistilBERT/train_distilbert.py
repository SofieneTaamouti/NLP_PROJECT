import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# =====================
# Paths
# =====================

TRAIN_PATH = "clean_data/Kaggle/kaggle_train.csv"
VALIDATION_PATH = "clean_data/Kaggle/kaggle_validation.csv"

OUTPUT_DIR = "outputs/distilbert_model"

TEXT_COL = "text"
LABEL_COL = "generated"   # colonne dans tes CSV : 0 = human, 1 = AI


# =====================
# Config CPU
# =====================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5

USE_SUBSAMPLE = False
N_TRAIN_PER_CLASS = 5000
N_VAL_PER_CLASS = 1000


# =====================
# Load and prepare data
# =====================

def load_and_prepare_data(path, n_per_class=None):
    df = pd.read_csv(path)

    print(f"\nLoaded file: {path}")
    print("Original columns:", df.columns.tolist())
    print("Original shape:", df.shape)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found. Available columns: {df.columns.tolist()}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Column '{LABEL_COL}' not found. Available columns: {df.columns.tolist()}")

    df = df[[TEXT_COL, LABEL_COL]].copy()

    blank_texts = df[TEXT_COL].isna() | (df[TEXT_COL].astype(str).str.strip() == "")
    blank_labels = df[LABEL_COL].isna() | (df[LABEL_COL].astype(str).str.strip() == "")

    print("Blank texts:", blank_texts.sum())
    print("Blank labels:", blank_labels.sum())

    df = df[~blank_texts & ~blank_labels].copy()

    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    print("Label distribution before subsampling:")
    print(df[LABEL_COL].value_counts())

    if USE_SUBSAMPLE and n_per_class is not None:
        sampled_parts = []

        for label_value in sorted(df[LABEL_COL].unique()):
            subset = df[df[LABEL_COL] == label_value]
            sampled_subset = subset.sample(
                n=min(len(subset), n_per_class),
                random_state=42
            )
            sampled_parts.append(sampled_subset)

        df = pd.concat(sampled_parts, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Columns after subsampling:", df.columns.tolist())

    print("Label distribution after subsampling:")
    print(df[LABEL_COL].value_counts())

    df = df.rename(columns={LABEL_COL: "labels"})

    print("Final columns:", df.columns.tolist())
    print("Final shape:", df.shape)

    return df

train_df = load_and_prepare_data(
    TRAIN_PATH,
    n_per_class=N_TRAIN_PER_CLASS
)

val_df = load_and_prepare_data(
    VALIDATION_PATH,
    n_per_class=N_VAL_PER_CLASS
)


# =====================
# Convert to Hugging Face Dataset
# =====================

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
val_dataset = Dataset.from_pandas(val_df, preserve_index=False)

print("\nHF train columns before tokenization:", train_dataset.column_names)


# =====================
# Tokenization
# =====================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(batch):
    return tokenizer(
        batch[TEXT_COL],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


train_dataset = train_dataset.map(tokenize_batch, batched=True)
val_dataset = val_dataset.map(tokenize_batch, batched=True)

# Remove raw text, keep labels
train_dataset = train_dataset.remove_columns([TEXT_COL])
val_dataset = val_dataset.remove_columns([TEXT_COL])

print("HF train columns after tokenization:", train_dataset.column_names)

if "labels" not in train_dataset.column_names:
    raise ValueError("Column 'labels' missing from train_dataset after tokenization.")

if "labels" not in val_dataset.column_names:
    raise ValueError("Column 'labels' missing from val_dataset after tokenization.")

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)


# =====================
# Model
# =====================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)


# =====================
# Metrics
# =====================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": f1,
    }


# =====================
# Training arguments
# =====================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_ai",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    use_cpu=True
)


# =====================
# Trainer
# =====================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


# =====================
# Train
# =====================

trainer.train()


# =====================
# Save model
# =====================

os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to: {OUTPUT_DIR}")