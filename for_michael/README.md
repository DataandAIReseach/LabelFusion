# LabelFusion-TS — paper package

Everything needed to reproduce and extend the draft in `paper/acl_latex.pdf`:
LabelFusion extended with a market–time-series expert, evaluated on hawkish/dovish
classification of Fed communication (Shah et al., ACL 2023), trained on sentences
up to 2015 and tested on 2015–2022.

## Layout

```
paper/        LaTeX source + compiled PDF (build: latexmk -pdf acl_latex.tex)
data/         gold_dated.csv (dated benchmark sentences), FRED series, LLM votes
artifacts/    trained-expert outputs (text expert, market expert) used by the paper
scripts/      the pipeline, numbered in dependency order
```

## Quick start — reproduce Table 1 in seconds (CPU only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/5_fuse.py
```

This trains only the small voting MLP on the shipped expert outputs and prints the
paper's results table.

## Regenerating each piece

| step | script | needs | time |
|---|---|---|---|
| 1 | `1_date_sentences.py` | clone of [gtfintechlab/fomc-hawkish-dovish](https://github.com/gtfintechlab/fomc-hawkish-dovish) | ~3 min |
| 2 | `2_collect_votes.py` | any OpenAI-compatible LLM endpoint (`LLM_API_KEY`, defaults: Ollama cloud, `gemma4:31b`) | ~10 min |
| 3 | `3_train_text_expert.py` | GPU recommended (Kaggle free T4 works; ~8 min) | 8 min GPU |
| 4 | `4_train_ts_expert.py` | CPU is fine | ~15 min |
| 5 | `5_fuse.py` | outputs of 1–4 (all shipped) | seconds |

Steps 1–4 write into `data/` and `artifacts/`; each is independent given its inputs, so
you can regenerate any single piece (e.g. swap the LLM in step 2, or the encoder in
step 3 via `--model roberta-base`) and rerun step 5.

## Design notes (matching the paper)

- **Time split**: train ≤ Sep 2015, test after — the 75% date quantile of the dated
  sentences; the most recent 15% of training dates serve as validation for epoch/model
  selection. No test-era information is used anywhere (market windows end the day
  before each document; CPI/unemployment enter with a 45-day publication lag).
- **Experts are trained individually and frozen**; only the small voting MLP sees them
  together (the LabelFusion recipe).
- **Market expert** (`ts_expert_med_pre.npz`): patch transformer, 18 weekly patches of a
  126-day × 6-series window, width 64, ~0.4M params; stage-0 pretrained on 1962–2015
  windows to predict near-term policy-rate changes, then trained window → stance.
- **Reported numbers are single training runs** (as in the benchmark paper and
  LabelFusion). Expect a few points of variation across seeds — a seed analysis exists
  and will be added to the paper before submission.

## Data provenance and licenses

- Benchmark sentences/labels: Shah, Paturi & Chava (ACL 2023), CC BY-NC 4.0, via their
  repository (only the derived `gold_dated.csv` is included here).
- Market series: FRED (public); the CSVs ship as downloaded
  (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>`).
- `llm_votes_fomc.json`: gemma4:31b outputs under the benchmark authors' verbatim prompt.

## Extension pointers

- New tasks: anything with dated financial text — steps 3–5 are task-agnostic given a
  `gold_dated.csv`-shaped file (`sentence,label,date`).
- New market channels: add series to `data/` and extend `market_window()` in
  `4_train_ts_expert.py` (VIX = `VIXCLS`, credit spreads = `BAA10Y`, ...).
- Stronger LLM expert: set `LLM_MODEL`/`LLM_API_URL` in step 2 — everything downstream
  is unchanged.
