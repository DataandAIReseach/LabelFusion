"""Add the exact document date (date, month, day) to every file in
data/training_data and data/test_data.

The benchmark files only carry a `year`. The dates come from the benchmark's own
source repository (gtfintechlab/fomc-hawkish-dovish), whose documents are named by
date (e.g. 20150917.txt). Each sentence is looked up in those documents:

  1. exact match (normalised) against the per-document sentence files, then
  2. containment match against the raw document text.

Candidates are narrowed by the document type taken from the file name
(lab-manual-mm-* = meeting minutes, pc = press conference, sp = speech; combine files
may match any type). The files' own `year` column is NOT used: it does not match the
year of the document the sentence comes from (only ~6% of rows agree). If exactly one
date remains the row gets it. Fallbacks (date_status tells them apart):

  * ambiguous (several possible dates): one is sampled at random from the candidates
    (status "sampled"; seeded, and a sentence with the same candidate list gets the same
    date in every file);
  * not found in any document: 1 January of the row's `year` column (status
    "year_fallback"). Caveat: that column often differs from the true document year.

New columns: date (YYYY-MM-DD), month, day, date_status, date_candidates.
Output: data/training_data_dated/ and data/test_data_dated/, same file names.
Nothing in the original files is changed.

    python testing/add_dates.py                       # clones the repo if needed
    python testing/add_dates.py --repo /path/to/fomc-hawkish-dovish
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import random

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path.home() / ".cache" / "fomc-hawkish-dovish"
SOURCE_URL = "https://github.com/gtfintechlab/fomc-hawkish-dovish"
SPLITS = ["training_data", "test_data"]
SEED = 5768

norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())


def doc_type(path: str) -> str | None:
    """mm / pc / sp from a source-repo path."""
    if "meeting_minutes" in path:
        return "mm"
    if "press_conference" in path:
        return "pc"
    if "speech" in path:
        return "sp"
    return None


def file_type(name: str) -> str | None:
    """Document type a benchmark file is restricted to (None for combine files)."""
    m = re.match(r"lab-manual-(mm|pc|sp)-", name)
    return m.group(1) if m else None


def build_indexes(source: Path):
    """exact: norm(sentence) -> {(date, type)};  raw: [(date, type, norm(text))]"""
    exact = defaultdict(set)
    for f in glob.glob(str(source / "data/filtered_data/**/*.csv"), recursive=True):
        m, t = re.search(r"(\d{8})", os.path.basename(f)), doc_type(f)
        if not m or not t:
            continue
        try:
            x = pd.read_csv(f)
        except Exception:
            continue
        col = "sentence" if "sentence" in x.columns else x.columns[1]
        for s in x[col].dropna():
            exact[norm(s)].add((m.group(1), t))

    raw = []
    for f in glob.glob(str(source / "data/raw_data/**/*.txt"), recursive=True):
        m, t = re.match(r"(\d{8})", os.path.basename(f)), doc_type(f)
        if not m or not t:
            continue
        raw.append((m.group(1), t, norm(open(f, encoding="utf-8", errors="replace").read())))
    return exact, raw


def resolve(key, exact, raw):
    """key = (normalised sentence, type or None) -> (status, candidate dates)."""
    n, typ = key
    ok = lambda t: typ is None or t == typ

    dates = {d for d, t in exact.get(n, ()) if ok(t)}
    how = "exact"
    if not dates:
        dates = {d for d, t, text in raw if ok(t) and n in text}
        how = "contained"
    if len(dates) == 1:
        return how, sorted(dates)
    return ("ambiguous" if dates else "unmatched"), sorted(dates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    if not (args.repo / "data").exists():
        args.repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", SOURCE_URL, str(args.repo)], check=True)

    print("Indexing source documents...")
    exact, raw = build_indexes(args.repo)
    print(f"  {len(exact)} indexed sentences, {len(raw)} raw documents")

    files = [f for s in SPLITS for f in sorted((REPO_ROOT / "data" / s).glob("lab-manual-*.xlsx"))]
    frames = {f: pd.read_excel(f) for f in files}

    cache: dict = {}
    for f, df in frames.items():
        typ = file_type(f.name)
        for s in df["sentence"]:
            key = (norm(s), typ)
            if key not in cache:
                cache[key] = resolve(key, exact, raw)

    # One random pick per sentence, so the same sentence gets the same date in every file.
    rng = random.Random(SEED)
    sampled = {}
    for key in sorted(cache, key=lambda k: (k[0], k[1] or "")):
        status, cands = cache[key]
        if status == "ambiguous" and (key[0], tuple(cands)) not in sampled:
            sampled[(key[0], tuple(cands))] = rng.choice(cands)

    overall = Counter()
    for f, df in frames.items():
        typ = file_type(f.name)
        keys = [(norm(s), typ) for s in df["sentence"]]
        res = [cache[k] for k in keys]
        chosen, status = [], []
        for key, (st, cands), year in zip(keys, res, df["year"]):
            if len(cands) == 1:
                chosen.append(cands[0]), status.append(st)
            elif cands:  # ambiguous -> random pick, fixed per sentence
                chosen.append(sampled[(key[0], tuple(cands))]), status.append("sampled")
            else:  # not found -> 1 January of the year column
                chosen.append(f"{int(year)}0101" if pd.notna(year) else None), status.append("year_fallback")
        dated = pd.to_datetime(chosen, format="%Y%m%d")
        out = df.copy()
        out["date"] = dated.strftime("%Y-%m-%d")
        out["month"] = dated.month.astype("Int64")
        out["day"] = dated.day.astype("Int64")
        out["date_status"] = status
        out["date_candidates"] = ["|".join(pd.to_datetime(c, format="%Y%m%d").strftime("%Y-%m-%d")) for _, c in res]

        out_dir = f.parent.parent / f"{f.parent.name}_dated"
        out_dir.mkdir(exist_ok=True)
        out.to_excel(out_dir / f.name, index=False)
        overall.update(out["date_status"])
        counts = out["date_status"].value_counts().to_dict()
        print(f"{f.parent.name}/{f.name}: {len(out)} rows {counts}")

    total = sum(overall.values())
    print(f"\nTotal rows: {total}")
    for status, n in overall.most_common():
        print(f"  {status:10s} {n:6d} ({n / total:.1%})")


if __name__ == "__main__":
    main()
