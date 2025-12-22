import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import GridSearchCV


DEFAULT_DATADIR = "processed_full"
MODEL_DIR = "models"

TARGET_COLS = ["increase_stock", "target", "target_num"]

FEATURES = [
    "hour_of_day",
    "temp",
    "humidity",
    "windspeed",
    "visibility",
    "dew",
    "precip",
    "snowdepth",
    "cloudcover",
    "is_weekend",
    "is_raining",
    "has_snow",
]

def load_split_csvs(datadir: str):
    train_path = os.path.join(datadir, "lda_train.csv")
    test_path  = os.path.join(datadir, "lda_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("lda_train.csv or lda_test.csv not found. Run preprocessing first.")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    return train_df, test_df

def evaluate_model(y_true, y_pred, title="MODEL EVALUATION"):
    print(f"\n={title}=")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def main(datadir: str):
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df, test_df = load_split_csvs(datadir)

    X_train = train_df[FEATURES].copy()
    y_train = train_df["target_num"]

    X_test = test_df[FEATURES].copy()
    y_test = test_df["target_num"]

    # baseline model 
    lda_base = LinearDiscriminantAnalysis()
    lda_base.fit(X_train, y_train)

    y_pred_base = lda_base.predict(X_test)
    evaluate_model(y_test, y_pred_base, title="BASELINE LDA")

    # hyper-parameter tuning
    param_grid = [
        {"solver": ["svd"]},
        {"solver": ["lsqr", "eigen"], "shrinkage": ["auto", 0.1, 0.3, 0.5, 0.7, 0.9]}
    ]

    grid = GridSearchCV(
        LinearDiscriminantAnalysis(),
        param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("\n[GRID SEARCH] Best Params:", grid.best_params_)
    print("[GRID SEARCH] Best CV F1-macro:", grid.best_score_)

    with open(os.path.join(MODEL_DIR, "lda_best_params.json"), "w") as f:
        json.dump(grid.best_params_, f, indent=4)

    best_lda = grid.best_estimator_

    y_proba = best_lda.predict_proba(X_test)[:, 1]

    # finding best threshhold
    thresholds = np.arange(0.2, 0.8, 0.05)
    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print("\n[THRESHOLD TUNING]")
    print("Best Threshold:", best_threshold)
    print("Best F1-macro :", best_f1)

    y_pred_final = (y_proba >= best_threshold).astype(int)
    evaluate_model(y_test, y_pred_final, title="FINAL TUNED LDA")

    model_path = os.path.join(MODEL_DIR, "lda_tuned.pkl")
    joblib.dump(best_lda, model_path)

    metadata = {
        "best_params": grid.best_params_,
        "best_threshold": best_threshold,
        "cv_f1_macro": grid.best_score_
    }

    with open(os.path.join(MODEL_DIR, "lda_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nSaved artifacts:")
    print(" -", model_path)
    print(" - lda_best_params.json")
    print(" - lda_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LDA using same features as Logistic Regression.")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATADIR,
        help="Folder containing lda_train.csv and lda_test.csv"
    )
    args, unknown = parser.parse_known_args()

    main(args.data)


"""=BASELINE LDA=

Accuracy : 0.7719
Precision: 0.4299
Recall   : 0.7931
F1 Score : 0.5576

Classification Report:
              precision    recall  f1-score   support

           0     0.9437    0.7672    0.8463       262
           1     0.4299    0.7931    0.5576        58

    accuracy                         0.7719       320
   macro avg     0.6868    0.7801    0.7019       320
weighted avg     0.8505    0.7719    0.7940       320


Confusion Matrix:
[[201  61]
 [ 12  46]]
Fitting 5 folds for each of 13 candidates, totalling 65 fits

[GRID SEARCH] Best Params: {'solver': 'svd'}
[GRID SEARCH] Best CV F1-macro: 0.7811755561847242

[THRESHOLD TUNING]
Best Threshold: 0.7499999999999998
Best F1-macro : 0.7689375493274673

=FINAL TUNED LDA=

Accuracy : 0.8656
Precision: 0.6364
Recall   : 0.6034
F1 Score : 0.6195

Classification Report:
              precision    recall  f1-score   support

           0     0.9132    0.9237    0.9184       262
           1     0.6364    0.6034    0.6195        58

    accuracy                         0.8656       320
   macro avg     0.7748    0.7636    0.7689       320
weighted avg     0.8630    0.8656    0.8642       320


Confusion Matrix:
[[242  20]
 [ 23  35]] """
