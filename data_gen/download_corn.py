import yfinance as yf
import os

STARTS = ['1970-01-01', '2012-01-01']
END = '2026-06-12'
SYMBOL = 'ZC=F'
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_prices')
OUTDIR = os.path.abspath(OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)

for start in STARTS:
    print(f"Downloading {SYMBOL} from {start} to {END}...")
    df = yf.download(SYMBOL, start=start, end=END, auto_adjust=True, progress=False)
    print(f"{SYMBOL}: {len(df)} rows")
    outname = f"{SYMBOL}_{start}_{END}.csv"
    outpath = os.path.join(OUTDIR, outname)
    df.to_csv(outpath)
    print(f"Saved: {outpath}")
