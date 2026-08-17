"""
Load every CSV in data/articles/csvs (produced by convert_json_to_csvs.py),
and for each one reproduce the same 80/20 sample split used in
train_test_llm.py (same RANDOM_STATE/SAMPLE_SIZE/TRAIN_FRAC, so it's the
identical 4000/1000 split), then further split the 80% (train_df) into a
500-row val set and the remaining train set.

Saves into data/articles/splits/, one train/val/test triplet per source CSV,
named "<source csv stem>_train.csv" / "_val.csv" / "_test.csv":
  - <stem>_train.csv  (3500 rows -- the 80% minus the 500 val rows)
  - <stem>_val.csv    (500 rows -- sampled from the 80%)
  - <stem>_test.csv   (1000 rows -- the original held-out 20%)
"""

from pathlib import Path
import glob

import pandas as pd
from sklearn.model_selection import train_test_split

from convert_json_to_csvs import ARTICLES_DIR, CSVS_DIR

OUT_DIR = ARTICLES_DIR / "splits"

RANDOM_STATE = 7  # same seed as train_test_llm.py -> identical sample/split
SAMPLE_SIZE = 5000
TRAIN_FRAC = 0.8
VAL_SIZE = 500  # rows carved out of the 80% train split


def split_one(csv_path):
    df = pd.read_csv(csv_path)
    print(f"{csv_path.name}: {df.shape}")

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

    print(f"  train: {len(train_final_df)} rows")
    print(f"  val:   {len(val_df)} rows")
    print(f"  test:  {len(test_df)} rows")

    stem = csv_path.stem
    train_final_df.to_csv(OUT_DIR / f"{stem}_train.csv", index=False)
    val_df.to_csv(OUT_DIR / f"{stem}_val.csv", index=False)
    test_df.to_csv(OUT_DIR / f"{stem}_test.csv", index=False)

    print(f"  saved: {stem}_{{train,val,test}}.csv -> {OUT_DIR}\n")


def main():
    csv_files = sorted(Path(p) for p in glob.glob(f"{CSVS_DIR}/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {CSVS_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_files:
        split_one(csv_path)


if __name__ == "__main__":
    main()
