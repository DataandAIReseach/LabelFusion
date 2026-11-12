"""Builds one self-contained HTML dashboard (a "Tableau" of the repo's data): outputs/plots/data_dashboard.html

Tabs: labelled sentence datasets (data/training_data, data/test_data), the low-data regimes
(data/*_dated_low_data_regimes), the time series (data/TSs, TSs_macro, stock_prices) and the generated articles (data/articles).
The data is aggregated here and embedded in the page; the page itself needs no server and no internet.

    python dataset_scripts/make_data_dashboard.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "plots" / "data_dashboard.html"
LABELS = {0: "dovish", 1: "hawkish", 2: "neutral"}
MAX_POINTS = 1500  # per series: longer ones are resampled to monthly means


def sentences() -> list[dict]:
    rows = []
    for split, folder in (("train", "training_data"), ("test", "test_data")):
        for path in sorted((DATA / folder).glob("lab-manual-*.xlsx")):
            m = re.match(rf"lab-manual-(.+)-{split}-(\d+)$", path.stem)
            if not m:
                continue
            df = pd.read_excel(path)
            c = df["label"].map(LABELS).value_counts()
            rows.append({"dataset": m[1], "split": split, "seed": m[2], "n": len(df), **{v: int(c.get(v, 0)) for v in LABELS.values()}})
    return rows


def regimes() -> list[dict]:
    rows = []
    for split in ("train", "test"):
        for path in sorted((DATA / f"{'training' if split == 'train' else 'test'}_data_dated_low_data_regimes").glob("*.csv")):
            m = re.match(rf"lab-manual-(.+)-{split}-(\d+)_(\d+)$", path.stem)
            if not m:
                continue
            df = pd.read_csv(path, usecols=["label"])
            c = df["label"].map(LABELS).value_counts()
            rows.append({"dataset": m[1], "split": split, "seed": m[2], "percent": int(m[3]), "n": len(df),
                         **{v: int(c.get(v, 0)) for v in LABELS.values()}})
    return rows


def read_series(path: Path, date_col: str, value_col: str, label: str, group: str) -> dict | None:
    df = pd.read_csv(path, usecols=[date_col, value_col]).dropna()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna().sort_values(date_col)
    if df.empty:
        return None
    n = len(df)
    if n > MAX_POINTS:
        df = df.set_index(date_col)[value_col].resample("MS").mean().dropna().reset_index()
    return {"name": label, "group": group, "n": n, "t": [d.strftime("%Y-%m-%d") for d in df[date_col]],
            "v": [round(float(v), 4) for v in df[value_col]]}


def time_series() -> list[dict]:
    out = []
    for p in sorted((DATA / "TSs").glob("*.csv")):
        d = pd.read_csv(p, nrows=1).columns
        s = read_series(p, d[0], d[1], p.stem, "TSs")
        out.append(s) if s else None
    for p in sorted((DATA / "TSs_macro").glob("*.csv")):
        s = read_series(p, "Date", "value", p.stem, "TSs_macro")
        out.append(s) if s else None
    for p in sorted((DATA / "stock_prices").glob("*.csv")):
        cols = pd.read_csv(p, nrows=1).columns
        if list(cols[:2]) == ["Date", "Kurs"]:
            s = read_series(p, "Date", "Kurs", p.stem, "stock_prices")
            out.append(s) if s else None
    base = DATA / "stock_prices" / "baseline_all_commodities_seed7.csv"
    if base.exists():
        for col in ("gold_price", "silver_price", "price_oil", "price_gas"):
            s = read_series(base, "date", col, f"synthetic_{col}", "synthetic (baseline_all_commodities)")
            out.append(s) if s else None
    return out


def articles() -> dict:
    path = next(iter(sorted((DATA / "articles" / "csvs").glob("*.csv"))), None)
    if path is None:
        return {}
    df = pd.read_csv(path, usecols=["article_date", "commodity", "gold", "silver", "oil", "gas"])
    df["year"] = pd.to_datetime(df["article_date"]).dt.year
    per_year = {c: df.groupby("year")[c].apply(lambda s: int(s.astype(bool).sum())).to_dict() for c in ("gold", "silver", "oil", "gas")}
    splits = {p.stem: int(sum(1 for _ in open(p, encoding="utf-8")) - 1) for p in sorted((DATA / "articles" / "splits").glob("*.csv"))}
    return {"file": path.name, "n": len(df), "combos": df["commodity"].value_counts().to_dict(),
            "years": sorted(int(y) for y in df["year"].unique()), "per_year": {c: {str(k): v for k, v in d.items()} for c, d in per_year.items()},
            "splits": splits}


TEMPLATE = (Path(__file__).with_name("data_dashboard_template.html")).read_text(encoding="utf-8")


def main() -> None:
    payload = {"sentences": sentences(), "regimes": regimes(), "series": time_series(), "articles": articles()}
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB", {k: len(v) for k, v in payload.items()})


if __name__ == "__main__":
    main()
