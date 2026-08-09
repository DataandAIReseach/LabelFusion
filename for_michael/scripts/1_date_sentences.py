"""Rebuild data/gold_dated.csv from the benchmark's repository (only needed if you want to
re-derive the dated sentences; the shipped CSV is this script's output).

1. Clone the benchmark: git clone https://github.com/gtfintechlab/fomc-hawkish-dovish
2. python scripts/1_date_sentences.py /path/to/fomc-hawkish-dovish

Matching: exact normalized match against the per-document sentence files, then containment
matching against the raw document text. Sentences matching several dates (recurring
boilerplate) or none are dropped. Expected yield: ~73% of the 2,312 unique gold sentences.
"""
import glob, os, re, sys
from collections import Counter, defaultdict
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = sys.argv[1] if len(sys.argv) > 1 else exit("usage: 1_date_sentences.py /path/to/fomc-hawkish-dovish")
TR = f"{REPO}/training_data/test-and-training"
LAB = {0: "dovish", 1: "hawkish", 2: "neutral"}
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())

gold = pd.concat([pd.read_excel(f"{TR}/training_data/lab-manual-combine-train-5768.xlsx"),
                  pd.read_excel(f"{TR}/test_data/lab-manual-combine-test-5768.xlsx")])
gold = gold[["sentence", "year", "label"]].dropna(subset=["sentence"])
agg = gold.groupby(gold["sentence"].map(norm)).agg(
    sentence=("sentence", "first"), year=("year", "first"),
    nlab=("label", "nunique"), label=("label", "first"))
gold = agg[agg.nlab == 1].reset_index(drop=True)
print(f"unique gold sentences: {len(gold)}")

fidx, ftyp = defaultdict(set), defaultdict(set)
DIRS = [("meeting_minutes", "mm"), ("press_conference", "pc"),
        ("select", "sp"), ("non-select", "sp"), ("all", "sp")]
for dirn, dtp in DIRS:
    for f in glob.glob(f"{REPO}/data/filtered_data/{dirn}/*.csv"):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if not m: continue
        try: x = pd.read_csv(f)
        except Exception: continue
        col = "sentence" if "sentence" in x.columns else x.columns[1]
        for s in x[col].dropna():
            n = norm(s); fidx[n].add(m.group(1)); ftyp[n].add(dtp)

raw = []
for f in glob.glob(f"{REPO}/data/raw_data/**/*.txt", recursive=True):
    m = re.match(r"(\d{8})", os.path.basename(f))
    if not m: continue
    dtp = ("mm" if "meeting_minutes" in f else "pc" if "press_conference" in f else "sp")
    try: t = norm(open(f, encoding="utf-8", errors="replace").read())
    except Exception: continue
    raw.append((m.group(1), dtp, t))

rows, stats = [], Counter()
for _, r in gold.iterrows():
    n = norm(r["sentence"])
    dates, typs, how = fidx.get(n, set()), ftyp.get(n, set()), "exact"
    if not dates:
        hits = [(d, tp) for d, tp, t in raw if n in t]
        dates = {d for d, _ in hits}; typs = {tp for _, tp in hits}; how = "contain"
    if len(dates) == 1:
        rows.append({"sentence": r["sentence"], "label": int(r["label"]),
                     "date": list(dates)[0], "doctype": "|".join(sorted(typs)), "how": how})
        stats[f"dated_{how}"] += 1
    else:
        stats["dropped_multidate" if dates else "dropped_unmatched"] += 1
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
print(dict(stats), f"-> dated {len(df)}/{len(gold)} = {len(df)/len(gold):.1%}")
df.to_csv(f"{ROOT}/data/gold_dated.csv", index=False)
print(f"wrote {ROOT}/data/gold_dated.csv")
