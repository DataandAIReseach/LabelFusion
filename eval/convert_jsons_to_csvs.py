"""
Convert every articles JSON file in data/articles/jsons into a flattened CSV
saved to data/articles/csvs, same logic as the "Load data" cell in
eval_silver_gold_oil_gas.ipynb.
"""

from pathlib import Path
import json
import glob
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = REPO_ROOT / "data" / "articles"
JSONS_DIR = ARTICLES_DIR / "jsons"
CSVS_DIR = ARTICLES_DIR / "csvs"

ALL_COMMODITIES = ["gold", "silver", "oil", "gas"]


def _flatten_articles_json(json_path):
    """Flatten one articles JSON file's contents into a DataFrame."""
    with open(json_path) as f:
        articles = json.load(f)

    # Flatten metadata + headline + body into rows
    rows = []
    for art in articles:
        meta = art["metadata"]
        row = {
            "obs_id":           meta["obs_id"],
            "article_date":     meta["article_date"],
            "commodity":        meta["commodity"],
            "n_words_target":   meta["n_words_target"],
            "references_break": meta.get("references_break"),
            "themes":           ", ".join(meta.get("themes", [])),
            "decoys_named":     ", ".join(meta.get("decoys_named", [])),
            "model":            meta["model"],
            "headline":         art["headline"],
            "body":             art["body"],
        }

        # one-hot commodity flags, read straight from metadata["commodities"]
        for c in ALL_COMMODITIES:
            row[c] = meta["commodities"].get(c, 0)

        # per-commodity price columns, only populated for commodities that
        # were actually sampled (others stay NaN)
        for c in ALL_COMMODITIES:
            entry = meta["prices"].get(c)
            row[f"current_price_{c}"] = entry["current_price"] if entry else None
            row[f"prices_21d_{c}"]    = entry["prices_21d"]    if entry else None

        rows.append(row)

    df = pd.DataFrame(rows)

    # reorder: metadata cols, then one-hot flags, then per-commodity prices, then text
    meta_cols   = ["obs_id", "article_date", "commodity"]
    onehot_cols = ALL_COMMODITIES
    price_cols  = [f"{prefix}_{c}" for c in ALL_COMMODITIES
                  for prefix in ("current_price", "prices_21d")]
    other_cols  = ["references_break", "themes", "decoys_named", "n_words_target", "model"]
    text_cols   = ["headline", "body"]

    return df[meta_cols + onehot_cols + price_cols + other_cols + text_cols]


def build_dataframe_from_latest_json():
    """Load the most recently modified articles JSON (in data/articles/jsons)
    and flatten it into a DataFrame.

    Returns:
        (df, json_path): the flattened DataFrame and the source JSON path.
    """
    json_files = sorted(glob.glob(f"{JSONS_DIR}/*.json"), key=lambda p: Path(p).stat().st_mtime)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {JSONS_DIR}")

    json_path = json_files[-1]  # most recently created/modified file
    print(f"Loading: {json_path}")

    return _flatten_articles_json(json_path), json_path


def _print_sanity_check(df):
    # ── Sanity check: distribution across commodity combinations ──────
    combo_counts = df.apply(
        lambda r: "+".join(c for c in ALL_COMMODITIES if r[c] == 1), axis=1
    ).value_counts()
    print("Commodity combination distribution:")
    print(combo_counts)

    n_commodities = df[ALL_COMMODITIES].sum(axis=1)
    print(f"\nArticles by number of commodities: {n_commodities.value_counts().sort_index().to_dict()}")


def convert_all():
    """Convert every JSON file in data/articles/jsons into a CSV in data/articles/csvs."""
    json_files = sorted(glob.glob(f"{JSONS_DIR}/*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {JSONS_DIR}")

    CSVS_DIR.mkdir(parents=True, exist_ok=True)

    for json_path in json_files:
        print(f"Loading: {json_path}")
        df = _flatten_articles_json(json_path)

        out_path = CSVS_DIR / (Path(json_path).stem + ".csv")
        df.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")
        print(f"Shape: {df.shape}")
        print()
        _print_sanity_check(df)
        print()


def main():
    convert_all()


if __name__ == "__main__":
    main()
