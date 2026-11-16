"""Read a synthetic articles JSON into the table the LabelFusion scripts feed to their models.

The JSON is written by dataset_scripts/data_gen/article_creation.ipynb
(dataset_scripts/data/articles/jsons/*.json). Every article has its text in "body", its label in
"label_id" (0 dovish, 1 hawkish, 2 neutral -- the lab-manual coding of the FOMC data) and its date in
metadata["article_date"].

read_articles_json() turns it into the same table the FOMC scripts build from their Excel files
(with_dates() in predict_low_data_regimes.py): one row per text with

    sentence | label | dovish | hawkish | neutral | date

plus a few metadata columns (obs_id, filler, label_flipped, series_stance, topic, IN, meta_text) that the
models ignore but that allow results to be broken down by text type. No Excel or CSV files are written.

split_train_test() gives a stratified train / test split, low_data_regimes() the nested 10 %, 20 %, ...,
100 % regimes of a training table with the same algorithm as dataset_scripts/make_low_data_regimes.py.

    from json_dataset import read_articles_json, split_train_test, low_data_regimes
    df = read_articles_json()                       # newest JSON in dataset_scripts/data/articles/jsons
    train_df, test_df = split_train_test(df)        # 80 / 20, stratified by label
    regimes = low_data_regimes(train_df)            # {10: DataFrame, 20: DataFrame, ..., 100: DataFrame}

    python dataset_scripts/json_dataset.py [path/to/articles.json]   # summary of the table, the split and the regimes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
JSONS_DIR = REPO_ROOT / "dataset_scripts" / "data" / "articles" / "jsons"
sys.path.insert(0, str(Path(__file__).resolve().parent))  # dataset_scripts, also when imported from elsewhere
from make_low_data_regimes import stratified_order  # noqa: E402  (the regime order of the FOMC files)

LABEL_MAP = {0: "dovish", 1: "hawkish", 2: "neutral"}   # as in testing/old/test_textclassify_llm.py
TEXT_COLUMN = "sentence"
DATE_COLUMN = "date"
RANDOM_STATE = 5768                                       # the project seed (testing/old/test_textclassify_llm.py)
MODEL_COLUMNS = [TEXT_COLUMN, "label", *LABEL_MAP.values(), DATE_COLUMN]
METADATA_COLUMNS = ["obs_id", "filler", "label_flipped", "series_stance", "topic", "IN", "meta_text"]


def latest_json(folder: Path = JSONS_DIR) -> Path:
    """The most recently written articles JSON in `folder`."""
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)
    files = [p for p in files if not p.name.endswith("_failures.json")]
    if not files:
        raise FileNotFoundError(f"no articles JSON in {folder} -- run dataset_scripts/data_gen/article_creation.ipynb first")
    return files[-1]


def read_articles_json(path: str | Path | None = None, drop_meta_text: bool = False) -> pd.DataFrame:
    """Articles JSON -> table with MODEL_COLUMNS first, then METADATA_COLUMNS.

    path:            the JSON file; None = the newest one in dataset_scripts/data/articles/jsons
    drop_meta_text:  drop the few texts in which the model left notes about its own answer (metadata meta_text)
    """
    path = Path(path) if path is not None else latest_json()
    with open(path, encoding="utf-8") as f:
        articles = json.load(f)

    rows = []
    for art in articles:
        meta = art["metadata"]
        label = art.get("label_id", meta.get("label"))
        if label is None:
            raise ValueError(f"{path}: article obs_id={meta.get('obs_id')} has no label")
        rows.append({
            TEXT_COLUMN:     (art.get("body") or "").strip(),
            "label":         int(label),
            DATE_COLUMN:     pd.Timestamp(meta["article_date"]).strftime("%Y-%m-%d"),
            "obs_id":        int(meta["obs_id"]),
            "filler":        bool(meta.get("filler", False)),
            "label_flipped": bool(meta.get("label_flipped", False)),
            "series_stance": meta.get("series_stance", art.get("label")),
            "topic":         meta.get("commodity", ""),
            "IN":            int(meta.get("IN", 0)),
            "meta_text":     bool(meta.get("meta_text", False)),
        })
    df = pd.DataFrame(rows)

    bad = set(df["label"]) - set(LABEL_MAP)
    if bad:
        raise ValueError(f"{path}: unknown label ids {sorted(bad)} (expected {sorted(LABEL_MAP)})")
    empty = df[TEXT_COLUMN] == ""
    if empty.any():
        print(f"{path.name}: dropping {int(empty.sum())} articles with an empty text")
    keep = ~empty
    if drop_meta_text:
        keep &= ~df["meta_text"]
    df = df[keep]

    # the models match texts by sentence (LLM caches, embedding memo): a repeated text must carry one label only
    dup = df[df.duplicated(TEXT_COLUMN, keep=False)]
    if dup.groupby(TEXT_COLUMN)["label"].nunique().gt(1).any():
        print(f"{path.name}: some identical texts have different labels -- keeping the first of each")
    df = df.drop_duplicates(TEXT_COLUMN).sort_values("obs_id").reset_index(drop=True)

    for label_id, name in LABEL_MAP.items():                      # one-hot label columns, as _add_onehot_labels
        df[name] = (df["label"] == label_id).astype(int)
    return df[MODEL_COLUMNS + METADATA_COLUMNS]


def split_train_test(df: pd.DataFrame, test_size: float = 0.2, seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train / test split (by label); both tables keep their row order by obs_id."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return (train_df.sort_values("obs_id").reset_index(drop=True),
            test_df.sort_values("obs_id").reset_index(drop=True))


def low_data_regimes(train_df: pd.DataFrame, percents=range(10, 101, 10), seed: int = RANDOM_STATE) -> dict[int, pd.DataFrame]:
    """Nested, label-stratified regimes of a training table: {percent: DataFrame}.

    Same algorithm as dataset_scripts/make_low_data_regimes.py: every regime is a prefix of one stratified
    order, so the 20 % regime contains the 10 % one and so on; a regime of p % has round(p/100 * n) rows."""
    order = stratified_order(train_df["label"].reset_index(drop=True), seed)
    regimes = {}
    for p in percents:
        n = max(1, round(len(train_df) * p / 100))
        regimes[int(p)] = train_df.iloc[np.sort(order[:n])].reset_index(drop=True)
    return regimes


def main() -> None:
    df = read_articles_json(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"{len(df)} texts | labels " + ", ".join(f"{n} {int((df['label'] == i).sum())}" for i, n in LABEL_MAP.items())
          + f" | filler {int(df['filler'].sum())} | flipped labels {int(df['label_flipped'].sum())}"
          + f" | dates {df[DATE_COLUMN].min()} .. {df[DATE_COLUMN].max()}")
    print(df[MODEL_COLUMNS].head(3).to_string(max_colwidth=60))
    train_df, test_df = split_train_test(df)
    print(f"\nsplit: train {len(train_df)} / test {len(test_df)}")
    regimes = low_data_regimes(train_df)
    print("regimes: " + ", ".join(f"{p}% {len(r)}" for p, r in regimes.items()))


if __name__ == "__main__":
    main()
