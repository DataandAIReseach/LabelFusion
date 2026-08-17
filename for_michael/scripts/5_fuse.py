"""Reproduce Table 1 of the paper from the shipped data and artifacts. Runs in seconds on CPU.

    python scripts/5_fuse.py

Inputs (relative to the repo root):
    data/gold_dated.csv            dated benchmark sentences (see scripts/1_date_sentences.py)
    data/llm_votes_fomc.json       LLM expert votes (see scripts/2_collect_votes.py)
    artifacts/expert_rb_b2_chrono_1.npz   text expert outputs (see scripts/3_train_text_expert.py)
    artifacts/ts_expert_med_pre.npz       market expert outputs (see scripts/4_train_ts_expert.py)
"""
import hashlib, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sha = lambda t: hashlib.sha1(str(t).encode()).hexdigest()
vkey = lambda t: hashlib.sha1(("authors-verbatim-v1|" + str(t)).encode()).hexdigest()

dated = pd.read_csv(f"{ROOT}/data/gold_dated.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
cutoff = dated.date.quantile(0.75)
tr = (dated.date <= cutoff).to_numpy()
te = (dated.date > cutoff).to_numpy()
y = dated.label.to_numpy(dtype=np.int64)
hashes = np.array([sha(s) for s in dated.sentence])
print(f"train {tr.sum()} / test {te.sum()} sentences (time split at {cutoff.date()})")

def aligned(path, keys):
    z = np.load(path)
    hmap = {h: i for i, h in enumerate(z["hash"])}
    idx = np.array([hmap[h] for h in hashes])
    return tuple(np.asarray(z[k], dtype=np.float32)[idx] for k in keys)

H, P_rb = aligned(f"{ROOT}/artifacts/expert_rb_b2_chrono_1.npz", ("h", "p"))
M, P_ts = aligned(f"{ROOT}/artifacts/ts_expert_med_pre.npz", ("m", "p"))

votes = json.load(open(f"{ROOT}/data/llm_votes_fomc.json"))
CLS = {"DOVISH": 0, "HAWKISH": 1, "NEUTRAL": 2}
Z = np.zeros((len(dated), 3), dtype=np.float32)
for i, s in enumerate(dated.sentence):
    t = votes.get(vkey(s)) or ""
    first = t.strip().splitlines()[0].upper() if t.strip() else ""
    p = next((CLS[w] for w in ("HAWKISH", "DOVISH", "NEUTRAL") if w in first), None)
    if p is None:
        p = next((CLS[w] for w in ("HAWKISH", "DOVISH", "NEUTRAL") if w in t.upper()), None)
    if p is not None:
        Z[i, p] = 1.0

def score(name, pred):
    return {"model": name,
            "wF1 (%)": round(100 * f1_score(y[te], pred, average="weighted"), 1),
            "Acc (%)": round(100 * accuracy_score(y[te], pred), 1)}

def fuse(name, blocks, seed=1):
    X = np.hstack(blocks)
    sc = StandardScaler().fit(X[tr])
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, early_stopping=True,
                        random_state=seed).fit(sc.transform(X[tr]), y[tr])
    return score(name, mlp.predict(sc.transform(X[te])))

rows = [
    score("majority class", np.full(te.sum(), np.bincount(y[tr]).argmax())),
    score("market expert alone", P_ts[te].argmax(1)),
    fuse("LabelFusion (text + LLM)", [H, Z]),
    score("RoBERTa-large, fine-tuned", P_rb[te].argmax(1)),
    score("LLM zero-shot", Z[te].argmax(1)),
    fuse("LabelFusion-TS (ours)", [H, Z, M]),
]
print("\n" + pd.DataFrame(rows).to_string(index=False))
print("\nNote: single training runs (as in the benchmark and the LabelFusion paper); "
      "results vary by a few points across random seeds.")
