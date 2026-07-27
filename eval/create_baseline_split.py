"""
Load the articles data via the same JSON -> flattened DataFrame logic as
convert_json_to_csvs.py, reproduce the same 80/20 sample split used in
train_test_llm.py (same RANDOM_STATE/SAMPLE_SIZE/TRAIN_FRAC, so it's the
identical 4000/1000 split), then further split the 80% (train_df) into a
500-row val set and the remaining train set.

Saves into data/articles/baseline_split/:
  - baseline_train.csv  (3500 rows -- the 80% minus the 500 val rows)
  - baseline_val.csv    (500 rows -- sampled from the 80%)
  - baseline_test.csv   (1000 rows -- the original held-out 20%)
"""

from pathlib import Path
from sklearn.model_selection import train_test_split

from convert_json_to_csvs import build_dataframe_from_latest_json, ARTICLES_DIR

OUT_DIR = ARTICLES_DIR / "baseline_split"

RANDOM_STATE = 7  # same seed as train_test_llm.py -> identical sample/split
SAMPLE_SIZE = 5000
TRAIN_FRAC = 0.8
VAL_SIZE = 500  # rows carved out of the 80% train split


def main():
    df, json_path = build_dataframe_from_latest_json()
    print(f"Full dataset shape: {df.shape}")

    # Same sampling/split as train_test_llm.py's load_sample_data()
    sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    train_df, test_df = train_test_split(
        sample_df, train_size=TRAIN_FRAC, random_state=RANDOM_STATE
    )

    # Carve VAL_SIZE rows out of the 80% train split; the rest stays as train
    val_df, train_final_df = train_test_split(
        train_df, train_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    train_final_df = train_final_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"baseline_train: {len(train_final_df)} rows")
    print(f"baseline_val:   {len(val_df)} rows")
    print(f"baseline_test:  {len(test_df)} rows")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_final_df.to_csv(OUT_DIR / "baseline_train.csv", index=False)
    val_df.to_csv(OUT_DIR / "baseline_val.csv", index=False)
    test_df.to_csv(OUT_DIR / "baseline_test.csv", index=False)

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
