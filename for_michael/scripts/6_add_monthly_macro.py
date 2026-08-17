"""Attach monthly macro values to gold_dated.csv.

For each labeled sentence, look up the value of every FRED series (see
for_michael/data/fred_*.csv table) for the sentence's year-month:
  - Monthly series (CPIAUCSL, UNRATE): the reported value for that month.
  - Daily series (DFF, DGS2, DGS10, NASDAQCOM): the mean of all daily
    observations within that month, so every series ends up on the same
    monthly grid.

Output: data/gold_dated_monthly_macro.csv (gold_dated.csv + one column per
FRED series, named after its series ID).

    python for_michael/scripts/6_add_monthly_macro.py
"""
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOR_MICHAEL_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(FOR_MICHAEL_ROOT)
TS_DIR = f"{REPO_ROOT}/data/TSs"

MONTHLY_SERIES = ["CPIAUCSL", "UNRATE"]
DAILY_SERIES = ["DFF", "DGS2", "DGS10", "NASDAQCOM"]


def load_monthly(series_id: str) -> pd.Series:
    """Load a FRED series and collapse it to one value per year-month.
    Already-monthly series (one row/month) pass through unchanged; daily
    series are averaged within each month."""
    df = pd.read_csv(f"{TS_DIR}/fred_{series_id}.csv", parse_dates=["observation_date"])
    df["year_month"] = df["observation_date"].dt.to_period("M")
    return df.groupby("year_month")[series_id].mean()


def main():
    dated = pd.read_csv(f"{FOR_MICHAEL_ROOT}/data/gold_dated.csv", parse_dates=["date"])
    dated["year_month"] = dated["date"].dt.to_period("M")

    for series_id in MONTHLY_SERIES + DAILY_SERIES:
        monthly = load_monthly(series_id)
        dated[series_id] = dated["year_month"].map(monthly)
        missing = dated[series_id].isna().sum()
        if missing:
            print(f"{series_id}: {missing} sentences with no matching month")

    out = dated.drop(columns=["year_month"])
    out_path = f"{FOR_MICHAEL_ROOT}/data/gold_dated_monthly_macro.csv"
    out.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(out)} rows, {len(MONTHLY_SERIES + DAILY_SERIES)} macro columns)")


if __name__ == "__main__":
    main()
