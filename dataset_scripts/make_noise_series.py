"""Uninformative control series: white noise, N(0, 1), i.i.d., one value per business day from 1970-01-01 to 2025-12-31.

Writes <out-dir>/NOISE1.csv ... NOISEn.csv (columns Date,value, same format as data/TSs_macro/) for use as
  python testing/predict_all_full_model.py --ts-dir data/TSs_noise --ts-series NOISE1 NOISE2 NOISE3 NOISE4 --out-tag noise
The series are independent of each other, of the dates and of the labels (seeded, so reproducible)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--start", default="1970-01-01")
parser.add_argument("--end", default="2025-12-31")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "TSs_noise")
args = parser.parse_args()

dates = pd.bdate_range(args.start, args.end)
rng = np.random.default_rng(args.seed)
args.out_dir.mkdir(parents=True, exist_ok=True)
for i in range(1, args.n + 1):
    pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "value": rng.standard_normal(len(dates)).round(4)}) \
        .to_csv(args.out_dir / f"NOISE{i}.csv", index=False)
print(f"{args.n} white-noise series x {len(dates)} business days ({dates[0].date()} .. {dates[-1].date()}) in {args.out_dir}")
