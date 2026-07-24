"""
Split the all-commodities article dataset into train/val/test (80/10/10).

Loads the already-flattened articles CSV (see data_gen.ipynb / the JSON-to-CSV
cell in eval_silver_gold_oil_gas.ipynb for how it was produced), splits it
80/10/10, and reports the commodity-combination sanity check for the full
dataset and each split, same convention as that notebook cell.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ARTICLES_DIR = Path(__file__).resolve().parent.parent / "data" / "articles"
CSV_PATH = ARTICLES_DIR / "articles_all_commodities_baseline_seed7_20260722_190748.csv"

ALL_COMMODITIES = ["gold", "silver", "oil", "gas"]
RANDOM_STATE = 7
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1


def print_combo_distribution(df: pd.DataFrame, name: str):
    combo_counts = df.apply(
        lambda r: "+".join(c for c in ALL_COMMODITIES if r[c] == 1), axis=1
    ).value_counts()
    print(f"\n{name} commodity combination distribution ({len(df)} rows):")
    print(combo_counts)

    n_commodities = df[ALL_COMMODITIES].sum(axis=1)
    print(f"{name} articles by number of commodities: {n_commodities.value_counts().sort_index().to_dict()}")


def main():
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Shape: {df.shape}")

    print_combo_distribution(df, "Full dataset")

    # 80/10/10: split off the 80% train first, then split the remaining 20%
    # evenly into val/test.
    train_df, rest_df = train_test_split(
        df, train_size=TRAIN_FRAC, random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        rest_df, train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC), random_state=RANDOM_STATE
    )

    print(f"\nSplit sizes -- train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print_combo_distribution(split_df, name)

    stem = CSV_PATH.stem
    print()
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = ARTICLES_DIR / f"{stem}_{name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}  Shape: {split_df.shape}")


if __name__ == "__main__":
    main()
