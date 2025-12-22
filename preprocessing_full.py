import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# CONFIG

DEFAULT_INPUT = "/teamspace/studios/this_studio/SML_PROJECT/training_data_ht2025.csv"
DEFAULT_OUTDIR = "processed_full"
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42

LABEL_COL = "increase_stock"

CATBOOST_CATEGORICALS = [
    "hour_of_day",
    "day_of_week",
    "month",
    "holiday",
    "weekday",
    "summertime",
    "is_weekend"
]


# FEATURE ENGINEERING (ORDER SAFE)

def feature_engineering(df: pd.DataFrame):
    df = df.copy()
    original_cols = list(df.columns)
    new_cols = []

    # Target mapping (ONLY if label exists)
    if LABEL_COL in df.columns:
        df["target_num"] = df[LABEL_COL].map({
            "high_bike_demand": 1,
            "low_bike_demand": 0
        })
        df["target"] = df["target_num"]
        new_cols += ["target_num", "target"]

    # Engineered features
    df["is_weekend"] = (df["day_of_week"].astype(int) >= 5).astype(int)
    df["is_raining"] = (df["precip"] > 0).astype(int)
    df["has_snow"] = (df["snowdepth"] > 0).astype(int)
    new_cols += ["is_weekend", "is_raining", "has_snow"]

    # Enforce column order
    df = df[original_cols + new_cols]

    return df, original_cols, new_cols


# HELPERS

def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")

def numeric_features(df, exclude):
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype != "object"
    ]


# MAIN PIPELINE

def main(input_csv, outdir, test_size):

    # -------------------------------
    # Load & feature engineering
    # -------------------------------
    df = pd.read_csv(input_csv)
    df, original_cols, engineered_cols = feature_engineering(df)

    save(df, f"{outdir}/base_full.csv")

    # -------------------------------
    # Train / test split
    # -------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["target_num"],
        random_state=RANDOM_STATE
    )

    save(train_df, f"{outdir}/base_train.csv")
    save(test_df, f"{outdir}/base_test.csv")

    
    # CATBOOST DATA 
    
    cat_train = train_df.copy()
    cat_test = test_df.copy()

    for c in CATBOOST_CATEGORICALS:
        if c in cat_train.columns:
            cat_train[c] = cat_train[c].astype(str)
            cat_test[c] = cat_test[c].astype(str)

    save(cat_train, f"{outdir}/catboost_train.csv")
    save(cat_test, f"{outdir}/catboost_test.csv")

    
    # SMOTE 
    
    label_cols = ["target_num", "target", LABEL_COL]
    num_cols = numeric_features(train_df, exclude=label_cols)

    X_train_num = train_df[num_cols]
    y_train = train_df["target_num"]

    smote = SMOTE(random_state=RANDOM_STATE)
    X_sm, y_sm = smote.fit_resample(X_train_num, y_train)

    # Reconstruct full dataframe (ORDER SAFE)
    smote_train = pd.DataFrame(columns=train_df.columns)
    smote_train[num_cols] = X_sm
    smote_train["target_num"] = y_sm
    smote_train["target"] = y_sm
    smote_train[LABEL_COL] = y_sm.map({
        1: "high_bike_demand",
        0: "low_bike_demand"
    })

    # Fill engineered flags using mode
    for col in engineered_cols:
        if col not in smote_train.columns:
            smote_train[col] = train_df[col].mode()[0]

    smote_train = smote_train[train_df.columns]

    
    # RANDOM FOREST DATA (SMOTE, NO SCALING)
    
    save(smote_train, f"{outdir}/rf_train.csv")
    save(test_df, f"{outdir}/rf_test.csv")

    
    # SCALING (LDA / LOGREG / OPTUNA)
    
    scaler = StandardScaler()

    scaled_train = smote_train.copy()
    scaled_test = test_df.copy()

    scaled_train[num_cols] = scaler.fit_transform(scaled_train[num_cols])
    scaled_test[num_cols] = scaler.transform(scaled_test[num_cols])

    save(scaled_train, f"{outdir}/lda_train.csv")
    save(scaled_test, f"{outdir}/lda_test.csv")

    save(scaled_train, f"{outdir}/logreg_train.csv")
    save(scaled_test, f"{outdir}/logreg_test.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified preprocessing (order-safe, SMOTE except CatBoost)."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    args = parser.parse_args()

    main(args.input, args.outdir, args.test_size)
