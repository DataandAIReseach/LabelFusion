"""Rewrite the IN texts of the struct / var / rho JSONs that are word-for-word identical to another text.

Haiku sometimes writes the same short sentence twice (mostly the content-free filler texts, whose prompt leaves little
room). A text counts as a duplicate when its body equals
  - any text of the baseline JSON (which stays untouched, its OUT texts are shared by every dataset), or
  - an IN text of a dataset that comes earlier (alphabetical order), or earlier in the same dataset.
The first occurrence is kept; every later one is written again with the prompt of the notebook
(dataset_scripts/article_creation.ipynb, cells a937b89f / d2ddf7e2: same model, effort, prompt builder and seed per
obs_id) plus a note listing the sentences it must not repeat. A new text that is again a duplicate (or contains model
notes) is requested again, up to --retries times.

OUT texts are never changed (they must stay identical to the baseline). Labels, stance, filler / flip flags stay as they
are -- they depend on obs_id and the series, not on the wording. Changed articles get metadata
"regenerated": true and "replaced_body": <old text>.

Before writing, the JSONs are copied to data/articles/jsons_backup_<timestamp>/.

    python dataset_scripts/regenerate_duplicate_texts.py            # dry run: list the duplicates, no API call
    python dataset_scripts/regenerate_duplicate_texts.py --run      # generate and write

--notes: rewrite instead the texts in which the model left its own notes (word counts "Count: The(1) ...", second
drafts "Wait, that is too long ..."). These also sit in the baseline's OUT texts, so this mode changes the baseline
JSON too: its texts are rewritten first, then every dataset's OUT text takes the baseline's new text (OUT stays
identical to the baseline), and the datasets' IN texts are rewritten one by one.

    python dataset_scripts/regenerate_duplicate_texts.py --notes          # dry run
    python dataset_scripts/regenerate_duplicate_texts.py --notes --run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
NOTEBOOK = SCRIPT_DIR / "article_creation.ipynb"
JSONS_DIR = REPO_ROOT / "data" / "articles" / "jsons"
BASELINE_JSON = REPO_ROOT / "data" / "articles_all_commodities_baseline_seed7_n5000_20261115_205224.json"
FULL_SERIES_DIR = REPO_ROOT / "data" / "stock_prices" / "full_series"
NOTEBOOK_CELLS = ("a937b89f", "d2ddf7e2")      # prompt builders / API call


def load_notebook() -> dict:
    """Run the prompt and API cells of the notebook (in its folder, for its relative paths) and return their names."""
    cells = {c.get("id"): "".join(c["source"]) for c in json.load(open(NOTEBOOK))["cells"]}
    ns: dict = {}
    cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    try:
        for cid in NOTEBOOK_CELLS:
            exec(compile(cells[cid], f"{NOTEBOOK.name}:{cid}", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns


def norm(text: str) -> str:
    return " ".join((text or "").split())


def find_duplicates(paths: list[Path]) -> tuple[dict[Path, list], dict[str, str]]:
    """{json: [(index, article, first occurrence)]} and {text: where it first appeared}."""
    first: dict[str, str] = {}
    for art in json.load(open(BASELINE_JSON)):
        first.setdefault(norm(art["body"]), "baseline")
    dups: dict[Path, list] = {}
    for path in paths:
        name = path.stem.removeprefix("articles_inflation_unemployment_").removesuffix("_seed7_n5000")
        for i, art in enumerate(json.load(open(path))):
            if art["metadata"]["IN"] != 1:
                continue
            key = norm(art["body"])
            if key in first:
                dups.setdefault(path, []).append((i, art, first[key]))
            else:
                first[key] = f"{name} #{i}"
    return dups, first


# Notes the model left in a text: its own word counts ("Count: The(1) Federal(2) ...", "Word count check: ..."), and
# second drafts ("Wait, that is too long ...", "Hmm, ..."). has_meta_text of the notebook misses most of them.
NOTES = re.compile(r"Count:|\w['.,;]?\(\d+\)|Word count|word-count|\bWait\b|\bHmm\b|Check:", re.I)


def has_notes(text: str) -> bool:
    return bool(NOTES.search(text or ""))


NOTE_START = re.compile(r"\b(Let me (?:count|recount|check|revise|rewrite|try|trim|shorten|fix)|Word count|Count:|Wait|Hmm|Check:|Revised:|Rewrite|Need about|That is \d+|That's \d+)", re.I)


def before_notes(text: str) -> str:
    """The sentences in front of the first note ("... decisions. Count: The(1) ..." -> "... decisions.")."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keep = []
    for s in sentences:
        if NOTE_START.match(s) or has_notes(s):
            break
        keep.append(s)
    return " ".join(keep).strip()


def find_notes(paths: list[Path]) -> dict[Path, list]:
    """{json: [(index, article)]} of the texts with model notes, the baseline first."""
    found = {}
    for path in [BASELINE_JSON, *paths]:
        items = [(i, a) for i, a in enumerate(json.load(open(path))) if has_notes(a["body"])]
        if items:
            found[path] = items
    return found


def series_file(path: Path) -> Path:
    if path == BASELINE_JSON:
        return FULL_SERIES_DIR / "full_baseline_inflation_unemployment_trial_seed7.csv"
    return FULL_SERIES_DIR / f"full_articles_{path.stem.removeprefix('articles_').removesuffix('_n5000')}.csv"


def avoid_note(texts: list[str]) -> str:
    if not texts:
        return ""
    lines = "\n".join(f"  - {t}" for t in texts)
    return ("\n\nAVOID REPETITION: other passages already use the wording below. Write a passage that is clearly "
            "different from each of them (different wording and sentence structure, not just a few swapped words):\n"
            + lines)


async def regenerate(nb: dict, path: Path, items: list, taken: set[str], retries: int) -> list[dict]:
    """New bodies for the given articles of one JSON. items: (index, article, origin[, sentences to avoid]);
    without the 4th element the article's own body is the sentence to avoid."""
    regime_df = None
    if any(not it[1]["metadata"]["filler"] for it in items):
        regime_df = nb["_load_baseline"](series_file(path))
    client = nb["_make_client"]()
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(nb["MAX_PARALLEL"])

    async def one(i, art, origin, avoid=None):
        m = art["metadata"]
        date, obs_id = m["article_date"], int(m["obs_id"])
        if m["filler"]:
            prompt = nb["build_filler_prompt"](article_date=date, seed=obs_id)
        else:
            parts = [c for c, v in m["commodities"].items() if v == 1]
            prices = {}
            for c in parts:
                p21, cur = nb["_get_prices"](regime_df, date, nb["PRICE_COLS"][c])
                prices[c] = {"prices_21d": p21, "current_price": cur}
            stance = nb["compute_stance"](regime_df, date, parts)    # the stance the text is written for
            prompt = nb["build_prompt"](article_date=date, commodities=parts, prices_by_commodity=prices,
                                        stance=stance, in_period=bool(m["IN"]), regime=m["regime"], seed=obs_id)
        avoid = [art["body"]] if avoid is None else list(avoid)
        for attempt in range(1, retries + 1):
            async with sem:
                raw, stop, model = await nb["_call_api"](client, {**prompt, "prompt": prompt["prompt"] + avoid_note(avoid)})
            body = " ".join(l.strip() for l in raw.split("\n") if l.strip())
            if has_notes(body):                         # keep the passage the model wrote before its notes
                body = before_notes(body)
            async with lock:
                ok = (len(body.split()) >= 4 and body[-1:] in ".!?" and norm(body) not in taken and not has_notes(body)
                      and not NOTE_START.search(body)
                      and len(body.split()) <= 3 * prompt["n_words"] + 15          # no second drafts
                      and not (m["filler"] and nb["count_policy_words"](body)))
                if ok:
                    taken.add(norm(body))
                    break
            if body and not has_notes(body):          # a text with notes is not worth quoting back
                avoid.append(body)
        else:
            print(f"  {path.name} #{i}: no new text after {retries} tries -- kept the old one")
            return None
        return {"index": i, "body": body, "old": art["body"], "origin": origin, "attempts": attempt,
                "model": model, "stop_reason": stop, "policy_words": nb["count_policy_words"](body)}

    results = await asyncio.gather(*(one(*it) for it in items))
    return [r for r in results if r]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="call the API and write the JSONs (default: dry run)")
    parser.add_argument("--retries", type=int, default=8, help="requests per text at most")
    parser.add_argument("--notes", action="store_true",
                        help="rewrite the texts with model notes (word counts, drafts) instead of the duplicates; "
                             "this one also changes the baseline JSON and the OUT texts")
    args = parser.parse_args()

    paths = sorted(p for p in JSONS_DIR.glob("articles_inflation_unemployment_*_n5000.json"))
    if args.notes:
        fix_notes(paths, args.run, args.retries)
        return
    dups, first = find_duplicates(paths)
    total = sum(len(v) for v in dups.values())
    print(f"{total} duplicate IN texts in {len(dups)} of {len(paths)} JSONs:")
    for path, items in dups.items():
        n_fill = sum(a["metadata"]["filler"] for _, a, _ in items)
        print(f"  {path.name}: {len(items)} ({n_fill} fillers)")
        for i, art, origin in items[:3]:
            print(f"      #{i} = {origin}: {art['body'][:90]!r}")
    if not args.run or not total:
        print("\ndry run -- nothing generated (use --run)" if total else "")
        return

    backup = JSONS_DIR.parent / f"jsons_backup_{datetime.now():%Y%m%d_%H%M%S}"
    backup.mkdir()
    for path in dups:
        shutil.copy2(path, backup / path.name)
    print(f"\nbackup: {backup}")

    nb = load_notebook()
    taken = set(first)
    for path, items in dups.items():
        new = asyncio.run(regenerate(nb, path, items, taken, args.retries))
        articles = json.load(open(path))
        for r in new:
            art = articles[r["index"]]
            m = art["metadata"]
            art["body"] = r["body"]
            m.update(regenerated=True, replaced_body=r["old"], duplicate_of=r["origin"],
                     n_words_actual=len(r["body"].split()), meta_text=False, attempts=r["attempts"],
                     policy_words_found=r["policy_words"], model=r["model"], stop_reason=r["stop_reason"])
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(articles, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)
        print(f"  {path.name}: {len(new)} of {len(items)} texts replaced")

    left, _ = find_duplicates(paths)
    print(f"\nduplicate IN texts left: {sum(len(v) for v in left.values())}")


def write_json(path: Path, articles: list) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(articles, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def apply(art: dict, r: dict, reason: str) -> None:
    m = art["metadata"]
    old = m.get("replaced_body", art["body"])          # keep the first original if a text is replaced twice
    art["body"] = r["body"]
    m.update(regenerated=True, regenerated_reason=reason, replaced_body=old,
             n_words_actual=len(r["body"].split()), meta_text=False, attempts=r["attempts"],
             policy_words_found=r["policy_words"], model=r["model"], stop_reason=r["stop_reason"])


def fix_notes(paths: list[Path], run: bool, retries: int) -> None:
    """Rewrite the texts with model notes: baseline first (IN and OUT), then the OUT texts of every dataset take the
    baseline's new text (so they stay identical to it), and the datasets' IN texts are written anew."""
    found = find_notes(paths)
    print("texts with model notes:")
    for path, items in found.items():
        n_in = sum(a["metadata"]["IN"] for _, a in items)
        print(f"  {path.name}: {len(items)} ({n_in} IN, {len(items) - n_in} OUT, "
              f"{sum(a['metadata']['filler'] for _, a in items)} fillers)")
    print("  e.g.", repr(next(iter(found.values()))[0][1]["body"][:160]))
    if not run or not found:
        print("\ndry run -- nothing generated (use --notes --run)" if found else "nothing to do")
        return

    backup = JSONS_DIR.parent / f"jsons_backup_{datetime.now():%Y%m%d_%H%M%S}"
    backup.mkdir()
    for path in found:
        shutil.copy2(path, backup / path.name)
    print(f"\nbackup: {backup}")

    bad = {norm(a["body"]) for items in found.values() for _, a in items}
    taken = {norm(a["body"]) for p in [BASELINE_JSON, *paths] for a in json.load(open(p))} - bad
    nb = load_notebook()

    # 1) baseline: every text with notes; the new OUT texts are passed on to the datasets
    out_map: dict[str, dict] = {}
    if BASELINE_JSON in found:
        items = [(i, a, "notes", []) for i, a in found[BASELINE_JSON]]
        articles = json.load(open(BASELINE_JSON))
        new = asyncio.run(regenerate(nb, BASELINE_JSON, items, taken, retries))
        for r in new:
            art = articles[r["index"]]
            if art["metadata"]["IN"] == 0:
                out_map[norm(art["body"])] = r
            apply(art, r, "model_notes")
        write_json(BASELINE_JSON, articles)
        print(f"  {BASELINE_JSON.name}: {len(new)} of {len(items)} texts replaced")

    # 2) datasets: OUT from the baseline, IN anew
    for path in paths:
        if path not in found:
            continue
        articles = json.load(open(path))
        out_items = [(i, a) for i, a in found[path] if a["metadata"]["IN"] == 0]
        in_items = [(i, a, "notes", []) for i, a in found[path] if a["metadata"]["IN"] == 1]
        n_out = 0
        for i, a in out_items:
            r = out_map.get(norm(a["body"]))
            if r is None:
                print(f"  {path.name} #{i}: OUT text not replaced in the baseline -- left as it is")
                continue
            apply(articles[i], {**r, "index": i}, "model_notes")
            n_out += 1
        new = asyncio.run(regenerate(nb, path, in_items, taken, retries)) if in_items else []
        for r in new:
            apply(articles[r["index"]], r, "model_notes")
        write_json(path, articles)
        print(f"  {path.name}: {n_out} of {len(out_items)} OUT (from the baseline), {len(new)} of {len(in_items)} IN replaced")

    # checks: no notes left, OUT texts still equal to the baseline's, no duplicate IN texts
    left = find_notes(paths)
    base_out = [a["body"] for a in json.load(open(BASELINE_JSON)) if a["metadata"]["IN"] == 0]
    same = all([a["body"] for a in json.load(open(p)) if a["metadata"]["IN"] == 0] == base_out for p in paths)
    dups, _ = find_duplicates(paths)
    print(f"\ntexts with notes left: {sum(len(v) for v in left.values())} | OUT texts identical to the baseline: {same} "
          f"| duplicate IN texts: {sum(len(v) for v in dups.values())}")


if __name__ == "__main__":
    main()
