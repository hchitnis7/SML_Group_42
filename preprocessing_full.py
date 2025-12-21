import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ==================================================
# CONFIG
# ==================================================
DEFAULT_INPUT = "/teamspace/studios/this_studio/SML_PROJECT/training_data_ht2025.csv"
DEFAULT_OUTDIR = "processed_full"
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET_COLS = ["increase_stock", "target", "target_num"]

CATBOOST_CATEGORICALS = [
    "hour_of_day",
    "day_of_week",
    "month",
    "holiday",
    "weekday",
    "summertime",
    "is_weekend"
]

# ==================================================
# FEATURE ENGINEERING
# ==================================================
def feature_engineering(df):
    df = df.copy()

    df["target_num"] = df["increase_stock"].map({
        "high_bike_demand": 1,
        "low_bike_demand": 0
    })
    df["target"] = df["target_num"]

    df["is_weekend"] = (df["day_of_week"].astype(int) >= 5).astype(int)
    df["is_raining"] = (df["precip"] > 0).astype(int)
    df["has_snow"] = (df["snowdepth"] > 0).astype(int)

    return df

# ==================================================
# HELPERS
# ==================================================
def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")

def numeric_features(df):
    return [
        c for c in df.columns
        if c not in TARGET_COLS and df[c].dtype != "object"
    ]

# MAIN PIPELINE
def main(input_csv, outdir, test_size):

    # Load & feature engineering
    
    df = pd.read_csv(input_csv)
    df = feature_engineering(df)

    
    # Train / test split
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["target_num"],
        random_state=RANDOM_STATE
    )

    save(df, f"{outdir}/base_full.csv")
    save(train_df, f"{outdir}/base_train.csv")
    save(test_df, f"{outdir}/base_test.csv")

    
    # SMOTE (numeric space)
    
    num_cols = numeric_features(train_df)

    X_train = train_df[num_cols]
    y_train = train_df["target_num"]

    smote = SMOTE(random_state=RANDOM_STATE)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)

    smote_train = pd.DataFrame(X_sm, columns=num_cols)
    smote_train["target_num"] = y_sm
    smote_train["target"] = y_sm
    smote_train["increase_stock"] = y_sm.map({
        1: "high_bike_demand",
        0: "low_bike_demand"
    })

    
    # CATBOOST
    
    cat_train = smote_train.copy()
    cat_test = test_df.copy()

    for c in CATBOOST_CATEGORICALS:
        if c in cat_train.columns:
            cat_train[c] = cat_train[c].astype(str)
            cat_test[c] = cat_test[c].astype(str)

    save(cat_train, f"{outdir}/catboost_train.csv")
    save(cat_test, f"{outdir}/catboost_test.csv")

    
    # RANDOM FOREST 
    # (SMOTE, NO scaling)
    
    save(smote_train, f"{outdir}/rf_train.csv")
    save(test_df, f"{outdir}/rf_test.csv")

    
    # SCALING 
    # (LDA + Logistic + Optuna)
    
    scaler = StandardScaler()

    scaled_train = smote_train.copy()
    scaled_test = test_df.copy()

    scaled_train[num_cols] = scaler.fit_transform(scaled_train[num_cols])
    scaled_test[num_cols] = scaler.transform(scaled_test[num_cols])

    
    # LDA
    
    save(scaled_train, f"{outdir}/lda_train.csv")
    save(scaled_test, f"{outdir}/lda_test.csv")

    
    # Logistic / Optuna
    
    save(scaled_train, f"{outdir}/logreg_train.csv")
    save(scaled_test, f"{outdir}/logreg_test.csv")

    print("\n[DONE] Preprocessing completed for ALL methods.")
    print("Model files now contain ZERO data preprocessing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified preprocessing for ALL models (SMOTE + scaling + categorical handling)."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    args = parser.parse_args()

    main(args.input, args.outdir, args.test_size)
