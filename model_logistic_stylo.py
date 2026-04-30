import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from stylometric_features import compute_stylometric_features


def train_kaggle_stylometric_logreg():
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
    X_train = X_train_text.apply(compute_stylometric_features).apply(pd.Series)
    X_validation = X_validation_text.apply(compute_stylometric_features).apply(pd.Series)
    X_test = X_test_text.apply(compute_stylometric_features).apply(pd.Series)

    # =========================================================
    # 3) Train logistic regression
    # =========================================================
    kaggle_stylometric_logreg = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    kaggle_stylometric_logreg.fit(X_train, y_train)

    # =========================================================
    # 4) Choose best threshold on validation set
    # =========================================================
    kaggle_validation_ai_probabilities = (
        kaggle_stylometric_logreg.predict_proba(X_validation)[:, 1]
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
    # 5) Evaluate on test set
    # =========================================================
    kaggle_test_ai_probabilities = (
        kaggle_stylometric_logreg.predict_proba(X_test)[:, 1]
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
    # 6) Inspect coefficients
    # =========================================================
    kaggle_feature_coefficients = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": kaggle_stylometric_logreg.coef_[0]
    })

    kaggle_top_ai_features = kaggle_feature_coefficients.sort_values(
        "coefficient", ascending=False
    )
    kaggle_top_human_features = kaggle_feature_coefficients.sort_values(
        "coefficient", ascending=True
    )

    print("\nTop stylometric features pushing toward AI:")
    print(kaggle_top_ai_features)

    print("\nTop stylometric features pushing toward human:")
    print(kaggle_top_human_features)

    # =========================================================
    # 7) Save model, threshold, and feature names
    # =========================================================
    output_dir = Path("outputs/saved_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        kaggle_stylometric_logreg,
        output_dir / "kaggle_stylometric_logreg.pkl"
    )
    joblib.dump(
        kaggle_best_threshold,
        output_dir / "kaggle_threshold_stylometric.pkl"
    )
    joblib.dump(
        list(X_train.columns),
        output_dir / "kaggle_stylometric_feature_names.pkl"
    )

    return {
        "model": kaggle_stylometric_logreg,
        "best_threshold": kaggle_best_threshold,
        "validation_results": kaggle_validation_results,
        "test_precision_ai": kaggle_test_precision_ai,
        "test_recall_ai": kaggle_test_recall_ai,
        "test_f1_ai": kaggle_test_f1_ai,
        "top_ai_features": kaggle_top_ai_features,
        "top_human_features": kaggle_top_human_features
    }


def predict_new_texts(texts):
    model_dir = Path("outputs/saved_models")

    model = joblib.load(model_dir / "kaggle_stylometric_logreg.pkl")
    threshold = joblib.load(model_dir / "kaggle_threshold_stylometric.pkl")
    feature_names = joblib.load(model_dir / "kaggle_stylometric_feature_names.pkl")

    X_new = pd.Series(texts).apply(compute_stylometric_features).apply(pd.Series)
    X_new = X_new.reindex(columns=feature_names, fill_value=0)

    probabilities = model.predict_proba(X_new)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame({
        "text": texts,
        "ai_probability": probabilities,
        "predicted_label": predictions
    })

    return results


if __name__ == "__main__":
    train_kaggle_stylometric_logreg()

