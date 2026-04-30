import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from stylometric_features import compute_stylometric_features


def train_kaggle_combined_logreg():
    # =========================================================
    # 1) Load clean Kaggle datasets
    # =========================================================
    data_dir = Path("clean_data/Kaggle")

    kaggle_train = pd.read_csv(data_dir / "kaggle_train.csv")
    kaggle_validation = pd.read_csv(data_dir / "kaggle_validation.csv")
    kaggle_test = pd.read_csv(data_dir / "kaggle_test.csv")

    X_train_text = kaggle_train["text"]
    y_train = kaggle_train["generated"]

    X_validation_text = kaggle_validation["text"]
    y_validation = kaggle_validation["generated"]

    X_test_text = kaggle_test["text"]
    y_test = kaggle_test["generated"]

    # =========================================================
    # 2) Compute stylometric features
    # =========================================================
    X_train_stylo = X_train_text.apply(compute_stylometric_features).apply(pd.Series)
    X_validation_stylo = X_validation_text.apply(compute_stylometric_features).apply(pd.Series)
    X_test_stylo = X_test_text.apply(compute_stylometric_features).apply(pd.Series)

    stylometric_feature_names = list(X_train_stylo.columns)

    # =========================================================
    # 3) Fit TF-IDF on training set only
    # =========================================================
    kaggle_tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train_tfidf = kaggle_tfidf.fit_transform(X_train_text)
    X_validation_tfidf = kaggle_tfidf.transform(X_validation_text)
    X_test_tfidf = kaggle_tfidf.transform(X_test_text)

    # =========================================================
    # 4) Combine TF-IDF + stylometric features
    # =========================================================
    X_train_combined = hstack([
        X_train_tfidf,
        csr_matrix(X_train_stylo.values)
    ])

    X_validation_combined = hstack([
        X_validation_tfidf,
        csr_matrix(X_validation_stylo.values)
    ])

    X_test_combined = hstack([
        X_test_tfidf,
        csr_matrix(X_test_stylo.values)
    ])

    # =========================================================
    # 5) Train logistic regression
    # =========================================================
    kaggle_combined_logreg = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    kaggle_combined_logreg.fit(X_train_combined, y_train)

    # =========================================================
    # 6) Choose best threshold on validation set
    # =========================================================
    kaggle_validation_ai_probabilities = (
        kaggle_combined_logreg.predict_proba(X_validation_combined)[:, 1]
    )

    kaggle_threshold_grid = np.arange(0.05, 0.96, 0.01)
    kaggle_validation_results = []

    for kaggle_threshold in kaggle_threshold_grid:
        kaggle_validation_predictions = (
            kaggle_validation_ai_probabilities >= kaggle_threshold
        ).astype(int)

        kaggle_validation_results.append({
            "threshold": kaggle_threshold,
            "precision_ai": precision_score(
                y_validation,
                kaggle_validation_predictions,
                pos_label=1,
                zero_division=0
            ),
            "recall_ai": recall_score(
                y_validation,
                kaggle_validation_predictions,
                pos_label=1,
                zero_division=0
            ),
            "f1_ai": f1_score(
                y_validation,
                kaggle_validation_predictions,
                pos_label=1,
                zero_division=0
            )
        })

    kaggle_validation_results = pd.DataFrame(kaggle_validation_results)

    kaggle_best_threshold_row = kaggle_validation_results.loc[
        kaggle_validation_results["f1_ai"].idxmax()
    ]
    kaggle_best_threshold = kaggle_best_threshold_row["threshold"]

    print("\nBest threshold selected on validation set:")
    print(kaggle_best_threshold_row)

    # =========================================================
    # 7) Evaluate on test set
    # =========================================================
    kaggle_test_ai_probabilities = (
        kaggle_combined_logreg.predict_proba(X_test_combined)[:, 1]
    )
    kaggle_test_predictions = (
        kaggle_test_ai_probabilities >= kaggle_best_threshold
    ).astype(int)

    kaggle_test_precision_ai = precision_score(
        y_test, kaggle_test_predictions, pos_label=1, zero_division=0
    )
    kaggle_test_recall_ai = recall_score(
        y_test, kaggle_test_predictions, pos_label=1, zero_division=0
    )
    kaggle_test_f1_ai = f1_score(
        y_test, kaggle_test_predictions, pos_label=1, zero_division=0
    )

    print("\nTest precision for AI:", kaggle_test_precision_ai)
    print("Test recall for AI:", kaggle_test_recall_ai)
    print("Test F1 for AI:", kaggle_test_f1_ai)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, kaggle_test_predictions))

    print("\nClassification report:")
    print(classification_report(y_test, kaggle_test_predictions, digits=3, zero_division=0))

    # =========================================================
    # 8) Save everything needed for reuse
    # =========================================================
    output_dir = Path("outputs/saved_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        kaggle_tfidf,
        output_dir / "kaggle_combined_tfidf_vectorizer.pkl"
    )
    joblib.dump(
        kaggle_combined_logreg,
        output_dir / "kaggle_combined_logreg.pkl"
    )
    joblib.dump(
        kaggle_best_threshold,
        output_dir / "kaggle_threshold_combined.pkl"
    )
    joblib.dump(
        stylometric_feature_names,
        output_dir / "kaggle_combined_stylometric_feature_names.pkl"
    )

    return {
        "tfidf_vectorizer": kaggle_tfidf,
        "model": kaggle_combined_logreg,
        "best_threshold": kaggle_best_threshold,
        "validation_results": kaggle_validation_results,
        "test_precision_ai": kaggle_test_precision_ai,
        "test_recall_ai": kaggle_test_recall_ai,
        "test_f1_ai": kaggle_test_f1_ai
    }


def predict_new_texts(texts):
    model_dir = Path("outputs/saved_models")

    tfidf_vectorizer = joblib.load(model_dir / "kaggle_combined_tfidf_vectorizer.pkl")
    model = joblib.load(model_dir / "kaggle_combined_logreg.pkl")
    threshold = joblib.load(model_dir / "kaggle_threshold_combined.pkl")
    stylometric_feature_names = joblib.load(
        model_dir / "kaggle_combined_stylometric_feature_names.pkl"
    )

    texts = pd.Series(texts)

    X_new_tfidf = tfidf_vectorizer.transform(texts)

    X_new_stylo = texts.apply(compute_stylometric_features).apply(pd.Series)
    X_new_stylo = X_new_stylo.reindex(columns=stylometric_feature_names, fill_value=0)

    X_new_combined = hstack([
        X_new_tfidf,
        csr_matrix(X_new_stylo.values)
    ])

    probabilities = model.predict_proba(X_new_combined)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame({
        "text": texts,
        "ai_probability": probabilities,
        "predicted_label": predictions
    })

    return results


if __name__ == "__main__":
    train_kaggle_combined_logreg()