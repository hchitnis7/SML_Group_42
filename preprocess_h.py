import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_INPUT = "/teamspace/studios/this_studio/SML_PROJECT/training_data_ht2025.csv"
DEFAULT_OUTDIR = "processed"
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42


def do_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform model-agnostic feature engineering.
    - Map target to numeric 'target_num'
    - Add is_weekend, is_raining, has_snow flags
    """
    df['target_num'] = df['increase_stock'].map({
        'high_bike_demand': 1,
        'low_bike_demand': 0
    })

    # Weekend flag (0=Mon .. 6=Sun)
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if int(x) >= 5 else 0)

    # Rare event flags
    df['is_raining'] = (df['precip'] > 0).astype(int)
    df['has_snow'] = (df['snowdepth'] > 0).astype(int)

    return df


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def main(input_csv: str, outdir: str, test_size: float):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Load
    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded {input_csv} ({df.shape[0]} rows, {df.shape[1]} cols)")

    df = do_feature_engineering(df)
    print("[INFO] Feature engineering completed. Columns now include:", ", ".join(df.columns))

    # Prepare output filenames
    base = os.path.basename(input_csv).replace(".csv", "")
    full_path = os.path.join(outdir, f"{base}_full_processed.csv")
    train_path = os.path.join(outdir, f"{base}_train.csv")
    test_path = os.path.join(outdir, f"{base}_test.csv")

    # Save full processed file
    save_csv(df, full_path)

    if 'target_num' not in df.columns:
        raise ValueError("target_num column missing after feature engineering.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df['target_num']
    )

    # Save splits
    save_csv(train_df.reset_index(drop=True), train_path)
    save_csv(test_df.reset_index(drop=True), test_path)

    # Print summary
    print(f"[INFO] Train/test split completed: train={len(train_df)}, test={len(test_df)} (test_size={test_size})")
    print(f"[INFO] Files saved in directory: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw bike-demand CSV and save train/test splits.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to raw CSV file.")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Directory to save processed CSVs.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Test set fraction (0-1).")
    args = parser.parse_args()

    main(args.input, args.outdir, args.test_size)
