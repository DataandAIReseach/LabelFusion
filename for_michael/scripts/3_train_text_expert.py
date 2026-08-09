"""Fine-tune the text expert (RoBERTa-large) and export its embeddings and scores.
Needs a GPU (~8 minutes on a T4; Kaggle free tier works). CPU is possible but slow (~2h).
Output: artifacts/expert_rb_b2_chrono_1.npz

    python scripts/3_train_text_expert.py [--model roberta-large] [--seed 1]
"""
import argparse, hashlib, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sha = lambda t: hashlib.sha1(str(t).encode()).hexdigest()


class Expert(nn.Module):
    def __init__(self, name, dim):
        super().__init__()
        self.rb = AutoModel.from_pretrained(name)
        self.head = nn.Linear(dim, 3)
    def forward(self, ids, mask):
        h = self.rb(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return h, self.head(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-large")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 1024 if "large" in args.model else 768
    MAXLEN, BATCH, ACC, LR, EPOCHS = 128, 16, 2, 1e-5, 6
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    dated = pd.read_csv(f"{ROOT}/data/gold_dated.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    cutoff = dated.date.quantile(0.75)
    tr_full = dated[dated.date <= cutoff]
    vcut = tr_full.date.quantile(0.85)
    tr = dated[dated.date <= vcut]; va = dated[(dated.date > vcut) & (dated.date <= cutoff)]
    print(f"device {DEV} | train {len(tr)} val {len(va)}")

    TOK = AutoTokenizer.from_pretrained(args.model)
    enc = lambda f: TOK(list(f.sentence.astype(str)), truncation=True, max_length=MAXLEN,
                        padding=True, return_tensors="pt")
    model = Expert(args.model, dim).to(DEV)
    w = torch.tensor(len(tr) / (3 * np.bincount(tr.label, minlength=3) + 1e-9),
                     dtype=torch.float32).to(DEV)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(enabled=(DEV == "cuda"))
    E, Ev = enc(tr), enc(va)
    y = torch.tensor(tr.label.to_numpy(dtype=np.int64))
    best = (-1, None)
    for ep in range(EPOCHS):
        model.train(); idx = np.random.permutation(len(y)); opt.zero_grad()
        for k, b0 in enumerate(range(0, len(idx), BATCH)):
            b = idx[b0:b0 + BATCH]
            with torch.amp.autocast(DEV, enabled=(DEV == "cuda")):
                _, out = model(E["input_ids"][b].to(DEV), E["attention_mask"][b].to(DEV))
                loss = lossf(out, y[b].to(DEV)) / ACC
            scaler.scale(loss).backward()
            if (k + 1) % ACC == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
        model.eval(); pv = []
        with torch.no_grad(), torch.amp.autocast(DEV, enabled=(DEV == "cuda")):
            for b0 in range(0, len(va), 64):
                _, o = model(Ev["input_ids"][b0:b0+64].to(DEV), Ev["attention_mask"][b0:b0+64].to(DEV))
                pv.append(o.float().argmax(1).cpu().numpy())
        f1v = f1_score(va.label, np.concatenate(pv), average="weighted")
        print(f"epoch {ep+1}/{EPOCHS} val wF1 {f1v:.4f}", flush=True)
        if f1v > best[0]:
            best = (f1v, {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()})
    model.load_state_dict(best[1]); model.eval()
    EA = enc(dated); Hs, Ps = [], []
    with torch.no_grad(), torch.amp.autocast(DEV, enabled=(DEV == "cuda")):
        for b0 in range(0, len(dated), 64):
            h, o = model(EA["input_ids"][b0:b0+64].to(DEV), EA["attention_mask"][b0:b0+64].to(DEV))
            Hs.append(h.float().cpu().numpy()); Ps.append(torch.softmax(o.float(), 1).cpu().numpy())
    out = f"{ROOT}/artifacts/expert_rb_b2_chrono_{args.seed}.npz"
    np.savez_compressed(out, hash=np.array([sha(s) for s in dated.sentence]),
                        h=np.concatenate(Hs).astype(np.float16),
                        p=np.concatenate(Ps).astype(np.float32))
    print(f"saved {out} (best val wF1 {best[0]:.4f})")


if __name__ == "__main__":
    main()
