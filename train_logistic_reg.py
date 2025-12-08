import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib


DEFAULT_DATADIR = "processed"
MODEL_DIR = "models"

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
    """
    datadir = folder containing:
        <name>_train.csv
        <name>_test.csv
    """
    files = [f for f in os.listdir(datadir) if f.endswith("_train.csv")]
    if len(files) == 0:
        raise FileNotFoundError(f"No *_train.csv found in {datadir}. Run preprocess.py first!")

    base = files[0].replace("_train.csv", "")
    train_path = os.path.join(datadir, f"{base}_train.csv")
    test_path = os.path.join(datadir, f"{base}_test.csv")

    print(f"[INFO] Loading:")
    print("  Train:", train_path)
    print("  Test :", test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def train_logistic_regression(X_train, y_train):
    """
    Fits a standard Logistic Regression classifier.
    """
    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced"     #since classes are imbalanced
    )
    model.fit(X_train, y_train)
    return model


def main(datadir: str):

    train_df, test_df = load_split_csvs(datadir)

    X_train = train_df[FEATURES].copy()
    y_train = train_df["target_num"]

    X_test = test_df[FEATURES].copy()
    y_test = test_df["target_num"]

    print(f"[INFO] Using {len(FEATURES)} features:")
    for f in FEATURES:
        print("  -", f)


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\n[INFO] Training Logistic Regression...")
    model = train_logistic_regression(X_train_scaled, y_train)
    print("[INFO] Training complete.")

    # Evaluate
    preds = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n====== EVALUATION RESULTS ======")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "logreg_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "logreg_scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print("\n[INFO] Saved model files:")
    print("  -", model_path)
    print("  -", scaler_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression on processed bike-demand data.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATADIR,
                        help="Folder containing *_train.csv and *_test.csv files.")
    args = parser.parse_args()

    main(args.data)


"""
====== EVALUATION RESULTS ======
Accuracy : 0.7781
Precision: 0.4414
Recall   : 0.8448
F1 Score : 0.5799

Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.76      0.85       262
           1       0.44      0.84      0.58        58

    accuracy                           0.78       320
   macro avg       0.70      0.80      0.71       320
weighted avg       0.86      0.78      0.80       320
"""