"""Low-data regimes of the dated files: 10%, 20%, ..., 100% of every file.

    data/training_data_dated/<name>.xlsx  ->  data/training_data_dated_low_data_regimes/
    data/test_data_dated/<name>.xlsx      ->  data/test_data_dated_low_data_regimes/

For every file (e.g. lab-manual-mm-train-5768) the script writes ten csv files to the output folder:

    <name>_10.csv  <name>_20.csv  ...  <name>_100.csv

The regimes are NESTED: the 20% file contains the rows of the 10% file, the 30% file those of the 20% file and
so on, so a growing regime only adds data. The rows are drawn STRATIFIED by label: the rows of every class are
shuffled (--seed) and spread evenly over a single order, so every prefix of that order, i.e. every regime,
keeps the class proportions of the whole file (up to rounding). A regime of p% has round(p/100 * n) rows;
the rows keep their original order and all columns of the xlsx file (sentence, label, date, ...).

    python dataset_scripts/make_low_data_regimes.py                          # training and test files
    python dataset_scripts/make_low_data_regimes.py --split train            # only the training files
    python dataset_scripts/make_low_data_regimes.py --split test --files lab-manual-mm-test-5768 --step 25
    python dataset_scripts/make_low_data_regimes.py --split train --seed 1 --out-dir /some/other/folder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDERS = {  # split -> (input folder, output folder)
    "train": (REPO_ROOT / "data" / "training_data_dated", REPO_ROOT / "data" / "training_data_dated_low_data_regimes"),
    "test": (REPO_ROOT / "data" / "test_data_dated", REPO_ROOT / "data" / "test_data_dated_low_data_regimes"),
}
LABEL_COLUMN = "label"


def stratified_order(labels: pd.Series, seed: int) -> np.ndarray:
    """Row positions in an order whose every prefix is (almost) stratified by label.

    Each class is shuffled; its j-th row (of n_c) gets the position (j + 0.5) / n_c in [0, 1]. Sorting all rows by
    that position interleaves the classes in their proportions."""
    rng = np.random.default_rng(seed)
    position = np.empty(len(labels))
    for label in labels.unique():
        rows = np.flatnonzero(labels.to_numpy() == label)
        rng.shuffle(rows)
        position[rows] = (np.arange(len(rows)) + 0.5) / len(rows)
    tie_break = rng.random(len(labels))  # classes with equal positions: random order
    return np.lexsort((tie_break, position))


def make_regimes(in_dir: Path, out_dir: Path, files: list[str] | None, step: int, seed: int) -> int:
    paths = sorted(in_dir.glob("*.xlsx"))
    if files:
        paths = [p for p in paths if p.stem in files]
    if not paths:
        raise SystemExit(f"no xlsx files found in {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    percents = list(range(step, 101, step))
    if percents[-1] != 100:
        percents.append(100)  # the full file is always the last regime

    for path in paths:
        df = pd.read_excel(path)
        order = stratified_order(df[LABEL_COLUMN], seed)
        sizes = []
        for percent in percents:
            n = max(1, round(len(df) * percent / 100))
            subset = df.iloc[np.sort(order[:n])]  # nested prefix of the order, original row order
            subset.to_csv(out_dir / f"{path.stem}_{percent}.csv", index=False)
            sizes.append(n)
        print(f"{path.stem}: {len(df)} rows -> {', '.join(f'{p}%={n}' for p, n in zip(percents, sizes))}")
    print(f"{len(paths) * len(percents)} files written to {out_dir}\n")
    return len(paths) * len(percents)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test", "both"), default="both",
                        help="which files to split: data/training_data_dated, data/test_data_dated or both (default)")
    parser.add_argument("--in-dir", type=Path, help="input folder instead of the default (needs --split train or test)")
    parser.add_argument("--out-dir", type=Path, help="output folder instead of the default (needs --split train or test)")
    parser.add_argument("--files", nargs="*", help="only these files (names without .xlsx) of the input folder")
    parser.add_argument("--step", type=int, default=10, help="size step in percent (default 10 -> 10, 20, ..., 100)")
    parser.add_argument("--seed", type=int, default=5768, help="seed of the shuffling (default: the project's 5768)")
    args = parser.parse_args()

    if not 1 <= args.step <= 100:
        raise SystemExit("--step must be between 1 and 100")
    if args.split == "both" and (args.in_dir or args.out_dir):
        raise SystemExit("--in-dir / --out-dir need --split train or --split test")

    for split in ("train", "test") if args.split == "both" else (args.split,):
        in_dir, out_dir = FOLDERS[split]
        make_regimes(args.in_dir or in_dir, args.out_dir or out_dir, args.files, args.step, args.seed)


if __name__ == "__main__":
    main()
