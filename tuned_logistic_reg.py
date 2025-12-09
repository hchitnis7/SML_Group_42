import argparse
import os
import json
import pandas as pd
import joblib
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from sklearn.model_selection import cross_val_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

DEFAULT_DATADIR = "/teamspace/studios/this_studio/processed"
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
    files = [f for f in os.listdir(datadir) if f.endswith("_train.csv")]
    if len(files) == 0:
        raise FileNotFoundError(f"No *_train.csv found in {datadir}. Run preprocess.py first!")

    base = files[0].replace("_train.csv", "")
    train_path = os.path.join(datadir, f"{base}_train.csv")
    test_path = os.path.join(datadir, f"{base}_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def evaluate_model(model, X_test, y_test, title="Model Evaluation"):
    preds = model.predict(X_test)

    print(f"\n={title} =")

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=4, zero_division=0))


def create_objective(X_train, y_train):
    def objective(trial):
        C = trial.suggest_loguniform("C", 1e-4, 1e2)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE()),
            ("model", LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=5000,
                class_weight="balanced"
            ))
        ])

        score = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="f1"
        ).mean()

        return score

    return objective


def main(datadir: str):
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df, test_df = load_split_csvs(datadir)

    X_train = train_df[FEATURES].copy()
    y_train = train_df["target_num"]

    X_test = test_df[FEATURES].copy()
    y_test = test_df["target_num"]

    # Baseline (no SMOTE, class_weight=balanced) 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    baseline_model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced"
    )
    baseline_model.fit(X_train_scaled, y_train)

    evaluate_model(baseline_model, X_test_scaled, y_test, title="BASELINE MODEL")

    #Optuna tuning with SMOTE 
    study = optuna.create_study(direction="maximize")
    objective = create_objective(X_train, y_train)
    study.optimize(objective, n_trials=40)

    print("\n[OPTUNA] Best Params:", study.best_params)
    print("[OPTUNA] Best F1 Score (CV):", study.best_value)

    with open(os.path.join(MODEL_DIR, "optuna_best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

    best_C = study.best_params["C"]

    # Final pipeline (scaler + SMOTE + logistic regression)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE()),
        ("model", LogisticRegression(
            C=best_C,
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)

    evaluate_model(pipeline, X_test, y_test, title="TUNED MODEL (SMOTE + OPTUNA)")

    model_path = os.path.join(MODEL_DIR, "logreg_optuna_smote.pkl")
    joblib.dump(pipeline, model_path)

    print("\nSaved model:")
    print("  -", model_path)
    print("  - optuna_best_params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression with SMOTE + Optuna.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATADIR,
                        help="Folder containing *_train.csv and *_test.csv files.")
    args = parser.parse_args()

    main(args.data)



"""====== BASELINE MODEL ======

Accuracy : 0.7781
Precision: 0.4414
Recall   : 0.8448
F1 Score : 0.5799

Classification Report:
              precision    recall  f1-score   support

           0     0.9569    0.7634    0.8493       262
           1     0.4414    0.8448    0.5799        58

    accuracy                         0.7781       320
   macro avg     0.6992    0.8041    0.7146       320
weighted avg     0.8635    0.7781    0.8004       320

====== TUNED MODEL (SMOTE + OPTUNA) ======

Accuracy : 0.7875
Precision: 0.4537
Recall   : 0.8448
F1 Score : 0.5904

Classification Report:
              precision    recall  f1-score   support

           0     0.9575    0.7748    0.8565       262
           1     0.4537    0.8448    0.5904        58

    accuracy                         0.7875       320
   macro avg     0.7056    0.8098    0.7235       320
weighted avg     0.8662    0.7875    0.8083       320"""