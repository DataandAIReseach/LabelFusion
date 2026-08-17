"""Train the market (time-series) expert and export its embeddings and scores.
CPU is fine (~10-15 minutes). Output: artifacts/ts_expert_med_pre.npz

    python scripts/4_train_ts_expert.py            # two-stage (pretrained), as in the paper
    python scripts/4_train_ts_expert.py --scratch  # skip stage-0 pretraining

Stage 0 (own objective, no labels needed): windows sampled every 2nd business day
1962..cutoff, target = does the policy rate change within 30 days (3-class).
Stage 1: window -> sentence stance label on the pre-cutoff training sentences.
"""
import argparse, hashlib, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)
T_WIN, PATCH = 126, 7
N_TOK = T_WIN // PATCH
sha = lambda t: hashlib.sha1(str(t).encode()).hexdigest()

SERIES = ("DFF", "DGS2", "DGS10", "NASDAQCOM", "CPIAUCSL", "UNRATE")
SER = {}
for name in SERIES:
    s = pd.read_csv(f"{ROOT}/data/fred_{name}.csv")
    s["observation_date"] = pd.to_datetime(s["observation_date"])
    SER[name] = s.set_index("observation_date")[name].apply(pd.to_numeric, errors="coerce").dropna()

def market_window(anchor):
    """126 business days x 6 channels; ends the day BEFORE anchor; CPI/UNRATE lagged 45 days."""
    end = anchor - pd.Timedelta(days=1)
    grid = pd.bdate_range(end=end, periods=T_WIN)
    def on_grid(s, lag=0):
        cut = end - pd.Timedelta(days=lag)
        g = s.loc[:cut].reindex(s.index.union(grid)).ffill().loc[grid].to_numpy(dtype=np.float64)
        return np.nan_to_num(g, nan=0.0)
    dff = on_grid(SER["DFF"]); g2 = on_grid(SER["DGS2"]); g10 = on_grid(SER["DGS10"])
    ndq = on_grid(SER["NASDAQCOM"])
    yoy = (SER["CPIAUCSL"] / SER["CPIAUCSL"].shift(12) - 1).dropna()
    yy = on_grid(yoy, 45); un = on_grid(SER["UNRATE"], 45)
    lnq = np.log(np.maximum(ndq, 1e-9))
    ch = np.stack([dff - dff[0], g2 - g2[0], (g10 - g2) - (g10[0] - g2[0]),
                   np.where(ndq[0] > 0, lnq - lnq[0], 0.0), (yy - yy[0]) * 100, un - un[0]], axis=1)
    return np.nan_to_num(ch).astype(np.float32)


class TSEncoder(nn.Module):
    """Small patch transformer: 18 weekly patches -> 64-dim market embedding."""
    def __init__(self, d=64, layers=4, heads=8, ff=128):
        super().__init__()
        self.proj = nn.Linear(PATCH * 6, d)
        self.pos = nn.Parameter(torch.zeros(1, N_TOK, d))
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 3)
    def embed(self, w):
        B = w.shape[0]
        x = w[:, :N_TOK * PATCH].reshape(B, N_TOK, PATCH * 6)
        return self.norm(self.enc(self.proj(x) + self.pos).mean(1))
    def forward(self, w):
        return self.head(self.embed(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", action="store_true", help="skip stage-0 pretraining")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    dated = pd.read_csv(f"{ROOT}/data/gold_dated.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    cutoff = dated.date.quantile(0.75)
    tr_full = dated[dated.date <= cutoff]
    vcut = tr_full.date.quantile(0.85)
    tr_m = (dated.date <= vcut).to_numpy()
    va_m = ((dated.date > vcut) & (dated.date <= cutoff)).to_numpy()
    y = dated.label.to_numpy(dtype=np.int64)
    model = TSEncoder()

    if not args.scratch:                                   # ── stage 0 ──
        dff = SER["DFF"]
        days = pd.bdate_range("1962-01-01", cutoff - pd.Timedelta(days=45))[::2]
        W, ya = [], []
        for d in days:
            fut = dff.loc[d:d + pd.Timedelta(days=30)]
            if not len(fut): continue
            delta = fut.iloc[-1] - dff.loc[:d].iloc[-1]
            ya.append(1 if delta > 0.10 else 0 if delta < -0.10 else 2)
            W.append(market_window(d))
        W = np.stack(W); ya = torch.tensor(np.array(ya))
        mu0, sd0 = W.mean((0, 1)), W.std((0, 1)) + 1e-8
        Wn = torch.tensor((W - mu0) / sd0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3); ce = nn.CrossEntropyLoss()
        print(f"stage 0: {len(Wn)} historical windows")
        for ep in range(15):
            idx = np.random.permutation(len(Wn))
            for b0 in range(0, len(idx), 256):
                b = torch.tensor(idx[b0:b0 + 256])
                opt.zero_grad(); ce(model(Wn[b]), ya[b]).backward(); opt.step()
            print(f"  stage-0 epoch {ep+1}/15", flush=True)

    # ── stage 1 ──
    Wtr = np.stack([market_window(d) for d in dated.date[tr_m]])
    mu, sd = Wtr.mean((0, 1)), Wtr.std((0, 1)) + 1e-8
    Wall = torch.tensor((np.stack([market_window(d) for d in dated.date]) - mu) / sd)
    Xtr, Xva = Wall[tr_m], Wall[va_m]
    ytr = torch.tensor(y[tr_m])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4); ce = nn.CrossEntropyLoss()
    best = (-1, None)
    for ep in range(40):
        model.train(); idx = np.random.permutation(len(Xtr))
        for b0 in range(0, len(idx), 128):
            b = torch.tensor(idx[b0:b0 + 128])
            opt.zero_grad(); ce(model(Xtr[b]), ytr[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            f1v = f1_score(y[va_m], model(Xva).argmax(1).numpy(), average="weighted")
        if f1v > best[0]:
            best = (f1v, {k: v.clone() for k, v in model.state_dict().items()})
        print(f"  stage-1 epoch {ep+1}/40 val wF1 {f1v:.4f}", flush=True)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        emb = model.embed(Wall).numpy().astype(np.float32)
        p = torch.softmax(model(Wall), 1).numpy().astype(np.float32)
    tag = "med_scratch" if args.scratch else "med_pre"
    out = f"{ROOT}/artifacts/ts_expert_{tag}.npz"
    np.savez_compressed(out, hash=np.array([sha(s) for s in dated.sentence]), m=emb, p=p)
    print(f"saved {out} (best val wF1 {best[0]:.4f})")


if __name__ == "__main__":
    main()
