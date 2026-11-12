"""Plots of all datasets in data/: the labelled sentence datasets (train/test, 3 seeds each) and the time series.

    python dataset_scripts/plot_datasets.py          # writes outputs/plots/datasets_sentences.png and datasets_time_series.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "plots"
LABELS = {0: "dovish", 1: "hawkish", 2: "neutral"}
COLORS = {"dovish": "#2a78d6", "hawkish": "#eb6834", "neutral": "#1baf7a"}  # categorical slots 1-3 of the reference palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def sentence_table() -> pd.DataFrame:
    rows = []
    for split, folder in (("train", "training_data"), ("test", "test_data")):
        for path in sorted((DATA / folder).glob("lab-manual-*.xlsx")):
            m = re.match(rf"lab-manual-(.+)-{split}-(\d+)$", path.stem)
            if not m:
                continue
            df = pd.read_excel(path)
            counts = df["label"].map(LABELS).value_counts()
            rows.append({"dataset": m[1], "split": split, "seed": m[2], "n": len(df),
                         **{lab: int(counts.get(lab, 0)) for lab in LABELS.values()}})
    return pd.DataFrame(rows)


def plot_sentences() -> Path:
    t = sentence_table()
    mean = t.groupby(["dataset", "split"])[["n", *LABELS.values()]].mean().reset_index()
    order = mean[mean.split == "train"].sort_values("n", ascending=False)["dataset"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.2]}, constrained_layout=True)
    y = range(len(order))
    for ax, split in zip(axes[:2], ("train", "test")):
        sub = mean[mean.split == split].set_index("dataset").loc[order]
        left = pd.Series(0.0, index=order)
        for lab in LABELS.values():
            ax.barh(list(y), sub[lab], left=left.values, color=COLORS[lab], label=lab, height=0.62, edgecolor="white", linewidth=1.5)
            left += sub[lab]
        for i, d in enumerate(order):
            ax.text(left[d] + sub.n.max() * 0.01, i, f"{sub.n[d]:.0f}", va="center", color=MUTED, fontsize=9)
        ax.set_yticks(list(y), order)
        ax.invert_yaxis()
        ax.set_title(f"{split}: sentences per dataset (mean of 3 seeds)", loc="left", fontsize=10.5)
        ax.set_xlim(0, sub.n.max() * 1.12)
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower right")
    ax = axes[2]  # label share in percent: train vs test
    share = mean.set_index(["dataset", "split"])
    h = 0.34
    for k, split in enumerate(("train", "test")):
        left = pd.Series(0.0, index=order)
        sub = share.xs(split, level="split").loc[order]
        for lab in LABELS.values():
            pct = sub[lab] / sub.n * 100
            ax.barh([i + (k - 0.5) * (h + 0.04) for i in y], pct, left=left.values, color=COLORS[lab], height=h,
                    alpha=1.0 if split == "train" else 0.55, edgecolor="white", linewidth=1.5)
            left += pct
    ax.set_yticks(list(y), order)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of sentences  (upper bar train, lower bar test)")
    ax.set_title("label share", loc="left", fontsize=10.5)
    fig.suptitle("Labelled sentence datasets (data/training_data, data/test_data; 8 datasets x 3 seeds)", x=0.01, ha="left", fontsize=12)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "datasets_sentences.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    t.to_csv(OUT / "datasets_sentences.csv", index=False)
    return path


def plot_time_series() -> Path:
    files = sorted((DATA / "TSs").glob("*.csv"))
    fig, axes = plt.subplots(2, 5, figsize=(16, 5.6), constrained_layout=True)
    for ax, path in zip(axes.flat, files):
        df = pd.read_csv(path)
        date, value = df.columns[0], df.columns[1]
        df[date] = pd.to_datetime(df[date])
        ax.plot(df[date], df[value], color="#2a78d6", linewidth=1.2)
        ax.set_title(f"{path.stem}  ({df[date].min():%Y}-{df[date].max():%Y}, n={len(df)})", loc="left", fontsize=9)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    fig.suptitle("Time series (data/TSs)", x=0.01, ha="left", fontsize=12)
    path = OUT / "datasets_time_series.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(plot_sentences())
    print(plot_time_series())
