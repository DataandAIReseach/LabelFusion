"""predict_all_labelFusion.py plus a time series branch: the three market series in data/3 Ts/daily_series.csv.

Everything of predict_all_labelFusion.py is unchanged (same splits, stage 1 roberta-large alone with grid + Optuna,
stage 2 fusion MLP on the frozen roberta-large embedding + the LLM label, same metrics and output layout). The
fusion MLP gets one more input: an embedding of what the market looked like on the day of the sentence.

  Data     data/test_data_dated/ and data/training_data_dated/ hold the same sentences, in the same order, as
           data/test_data/ and data/training_data/, plus the `date` of the statement. The splits are therefore
           the same as in predict_all_labelFusion.py (the sentences are checked to match the plain files).

  TS       DGS10 (10-year Treasury yield), T10Y2Y (10y-2y yield spread) and VIXCLS (VIX), daily, from
           data/3 Ts/daily_series.csv. Holidays (empty cells) are filled with the last known value.
           FIXED WINDOW: for a sentence dated d, every series contributes its last --ts-window (default 21)
           values up to and including d, the same window length for every sentence (textclassify's
           TS.get_series_batch with window_days). With --ts-window 0 the window is the whole series: from its
           beginning (1996-01-02) up to and including d, so it grows with the date (about 21 values for the
           earliest, about 7000 for the latest sentences). One TimesFM-2.5 encoder per series (pooled over
           the window) -> CrossTSTransformer (attention across the three series) -> a vector of --ts-dim
           (default 64) that is concatenated to [roberta-large embedding, LLM one-hot] in the fusion MLP.
           The TimesFM encoders are frozen by default (the CrossTSTransformer and the MLP are trained);
           --ts-finetune also trains them with the MLP's learning rate. Every Optuna trial and the final MLP
           start from the pretrained TimesFM weights. As frozen encoders always give the same embedding for the
           same window, their embeddings are computed once per (series, window) and reused across epochs,
           Optuna trials and the final MLP.

  Pooling  --ts-pool-only skips the CrossTSTransformer's attention and projection: the three per-series
           embeddings (1280 each) are only pooled over the series (--ts-pooling mean or max, CrossTSTransformer.pool)
           and the pooled vector goes straight into the fusion MLP, so the TS input of the MLP is 1280-dimensional
           instead of --ts-dim.

Reuse of stage 1: --roberta-params-from <summary_seeds.csv of predict_all_labelFusion.py> takes the chosen
learning rate / batch size / epochs of each pair from that run instead of tuning roberta-large again (the
fine-tuned roberta-large itself is not saved, so it is trained once more with these parameters). The result is
then comparable to the run without time series: the only difference is the TS branch.

Writes to ./outputs/labelfusion_roberta_large_ts_fixed_window/ (same files as predict_all_labelFusion.py).

Several runs (e.g. one sbatch job per group of datasets) can run at the same time without influencing each other.
Every run has its own work folder for the library's prediction caches (.work/<job id>) and its own folder for the
saved roberta-large models, and every result file carries the model and the start time of the run in its name
(<name>_ts_fixed_window_<YYYYMMDD_HHMMSS>.csv / .json), so no run overwrites the files of another:
    <test file>_ts_fixed_window_<date_time>.csv / .json        per pair
    <dataset>_summary_ts_fixed_window_<date_time>.json         per dataset
    summary_ts_fixed_window_<date_time>.csv, summary_seeds_ts_fixed_window_<date_time>.csv   per run
    trials/<test file>_trials_... .csv and _mlp_trials_... .csv
Each run's summary only holds the pairs of that run.

    python testing/old/predict_all_labelFusion_TS_fixed_window.py                                   # every dataset, every seed
    python testing/old/predict_all_labelFusion_TS_fixed_window.py --datasets mm-test pc-split-test
    python testing/old/predict_all_labelFusion_TS_fixed_window.py --roberta-params-from outputs/labelfusion_roberta_large/summary_seeds.csv
    python testing/old/predict_all_labelFusion_TS_fixed_window.py --ts-window 63 --ts-dim 128
    python testing/old/predict_all_labelFusion_TS_fixed_window.py --ts-pool-only                    # pooled embeddings, no attention
    python testing/old/predict_all_labelFusion_TS_fixed_window.py --ts-window 0                      # whole series up to the date
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import os
import shutil
import sys
import types
from datetime import datetime
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING_DIR))

# Everything this script writes lives in RESULT_DIR. Runs must not influence each other's results: the library names
# its saved roberta-large models after the train texts, so two runs on the same pair would write and delete the same
# directory. Every run therefore gets its own work folder (.work/<job id>) in RESULT_DIR for the models and the
# library's prediction caches; the folder is removed when the run ends. (Set before the modules below read it.)
RESULT_DIR = TESTING_DIR.parent / "outputs" / "labelfusion_roberta_large_ts_fixed_window"
RUN_ID = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
WORK_DIR = RESULT_DIR / ".work" / RUN_ID
os.environ["LABELFUSION_MODEL_CACHE"] = str(WORK_DIR / "models")
atexit.register(shutil.rmtree, WORK_DIR, ignore_errors=True)
SUFFIX = f"ts_fixed_window_{datetime.now():%Y%m%d_%H%M%S}"  # <model>_<date_time> of this run, part of every result file name

import predict_all_files_labelFusion as files  # noqa: E402
import predict_all_labelFusion as base  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from textclassify.timeseries import CrossTSTransformer, TS, TSEmbedder  # noqa: E402

stage1 = base.stage1
plain_load = base.load  # base.load is replaced by load_dated below
REPO_ROOT = base.TEST_DIR.parent.parent
DATED_TRAIN_DIR = REPO_ROOT / "data" / "training_data_dated"
DATED_TEST_DIR = REPO_ROOT / "data" / "test_data_dated"
TS_FILE = REPO_ROOT / "data" / "3 Ts" / "daily_series.csv"
TS_SERIES = ["DGS10", "T10Y2Y", "VIXCLS"]
DATE_COLUMN = "date"
FULL_HISTORY = 10**6  # window_days that reaches back to the beginning of every series



class DailySeries(TS):
    """textclassify's TS loader on one CSV with a DATE column and one column per series."""

    def __init__(self, csv_path: Path, series_names: list[str]):
        super().__init__(data_dir=csv_path.parent, stock_symbols=list(series_names), date_column="DATE", price_column="value")
        self.csv_path = csv_path

    def load_all(self) -> None:
        raw = pd.read_csv(self.csv_path, parse_dates=["DATE"]).sort_values("DATE")
        for name in self.stock_symbols:
            series = raw[["DATE", name]].rename(columns={name: "value"})
            series["value"] = series["value"].ffill()  # holidays: the last known value
            self.data[name] = series.dropna().reset_index(drop=True)


class CachedEmbedder(nn.Module):
    """A FROZEN TSEmbedder that remembers its embedding per input series. The same window always gives the same
    embedding, so epochs, Optuna trials and the final MLP only compute each (series, window) once; the cache is
    shared by all copies of the encoder (they start from the same pretrained weights)."""

    def __init__(self, inner: TSEmbedder, cache: dict):
        super().__init__()
        self.inner = inner
        self.cache = cache

    def forward(self, past_values, **kwargs):
        keys = [hashlib.blake2b(np.asarray(v, dtype=np.float32).tobytes(), digest_size=16).digest() for v in past_values]
        todo = {k: v for k, v in zip(keys, past_values) if k not in self.cache}
        if todo:
            with torch.no_grad():
                embeddings = self.inner(list(todo.values()))["embeddings"].detach().cpu()
            self.cache.update(zip(todo, embeddings))
        device = next(self.inner.parameters()).device
        return {"embeddings": torch.stack([self.cache[k] for k in keys]).to(device)}


def run_pooled_ts_branch(self, dates):
    """FusionEnsemble._run_ts_branch without the CrossTSTransformer's attention: the per-series embeddings are
    pooled over the series and passed on as they are."""
    tr = self.ts_transformer
    batch = self.ts_loader.get_series_batch(dates, window_days=self.ts_window_days, stocks=tr.series_names)
    embeddings = {name: self.ts_embedders[name](batch[name])["embeddings"].to(self.device) for name in tr.series_names}
    return tr.pool(embeddings, self.ts_pooling)["output"]


class TSBranch:
    """Loads the series and the pretrained TimesFM encoders once; fresh() hands every fusion MLP its own copy
    (a fine-tuned or trained branch must never leak into the next Optuna trial)."""

    def __init__(self, args):
        self.loader = DailySeries(TS_FILE, TS_SERIES)
        self.loader.load_all()
        self.encoders = {name: TSEmbedder(pooling="mean") for name in TS_SERIES}
        self.caches = {name: {} for name in TS_SERIES}  # embeddings of frozen encoders, per series
        self.hidden_size = next(iter(self.encoders.values())).hidden_size
        self.output_dim = args.ts_dim
        self.finetune = args.ts_finetune

    def fresh(self):
        encoders = {name: copy.deepcopy(enc) for name, enc in self.encoders.items()}
        if not self.finetune:
            for enc in encoders.values():
                enc.requires_grad_(False)
            encoders = {name: CachedEmbedder(enc, self.caches[name]) for name, enc in encoders.items()}
        torch.manual_seed(base.RANDOM_STATE)
        transformer = CrossTSTransformer(series_names=TS_SERIES, hidden_size=self.hidden_size, output_dim=self.output_dim)
        return self.loader, encoders, transformer


def dated_path(plain: Path) -> Path:
    return (DATED_TEST_DIR if plain.parent == base.TEST_DIR else DATED_TRAIN_DIR) / plain.name


def load_dated(plain: Path) -> pd.DataFrame:
    """sentence + label (+ one-hot) as base.load, plus the date of every sentence."""
    df = plain_load(plain)
    dated = pd.read_excel(dated_path(plain))
    if len(dated) != len(df) or not (dated[base.TEXT_COLUMN].values == df[base.TEXT_COLUMN].values).all():
        raise ValueError(f"{dated_path(plain).name} does not hold the same sentences in the same order as {plain.name}")
    df[DATE_COLUMN] = pd.to_datetime(dated[DATE_COLUMN]).dt.strftime("%Y-%m-%d").values
    return df


def split_dated(train_path: Path, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same split as base.split_pair (80/20 stratified, same seed), with the dates."""
    train_full = load_dated(train_path)
    if args.limit_train:
        train_full = train_full.sample(n=args.limit_train, random_state=base.RANDOM_STATE).reset_index(drop=True)
    train_df, val_df = train_test_split(
        train_full, train_size=0.8, random_state=base.RANDOM_STATE, stratify=train_full["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def publish(staging: Path) -> None:
    """Move what the run wrote into its staging folder to RESULT_DIR, with the model and the start time of the run
    in every file name: <name>_ts_fixed_window_<YYYYMMDD_HHMMSS>.<ext>. Runs never write the same file."""
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            target_dir = RESULT_DIR / path.parent.relative_to(staging)
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(path, target_dir / f"{path.stem}_{SUFFIX}{path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="only datasets (test file name without seed) containing one of these")
    parser.add_argument("--epochs", type=int, default=10,
                        help="roberta-large (maximum) epochs of the default configuration and of the 16 grid points")
    parser.add_argument("--retries", type=int, default=2,
                        help="repeat the final roberta-large training with another seed up to N times if it collapses to one class")
    parser.add_argument("--limit-train", type=int, help="subsample the train file to N rows (smoke test)")
    parser.add_argument("--trials", type=int, default=24,
                        help="stage-1 Optuna trials per pair: the 16 grid points first, the rest is free refinement "
                             "(0 = defaults)")
    parser.add_argument("--mlp-trials", type=int, default=30,
                        help="stage-2 Optuna trials per pair tuning the fusion MLP (0 = defaults)")
    parser.add_argument("--ml-proj-dim", type=int, default=768,
                        help="project roberta-large's 1024-dim [CLS] embedding down to this size with a "
                             "trainable Linear+ReLU before the fusion MLP; 0 disables the projection")
    parser.add_argument("--ts-window", type=int, default=21,
                        help="fixed window: number of daily values of every series up to the sentence's date; "
                             "0 = the whole series, from its beginning up to the date")
    parser.add_argument("--ts-dim", type=int, default=64, help="size of the time series embedding fed to the fusion MLP")
    parser.add_argument("--ts-pool-only", action="store_true",
                        help="no attention / projection: pool the per-series embeddings and pass them straight to the "
                             "fusion MLP (1280 inputs instead of --ts-dim)")
    parser.add_argument("--ts-pooling", choices=("mean", "max"), default="mean",
                        help="pooling over the series with --ts-pool-only")
    parser.add_argument("--ts-finetune", action="store_true",
                        help="also train the TimesFM encoders (default: frozen, only the CrossTSTransformer is trained)")
    parser.add_argument("--roberta-params-from", type=Path,
                        help="summary_seeds.csv of a predict_all_labelFusion.py run: reuse its roberta-large parameters "
                             "per pair instead of tuning stage 1 again")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    # FusionEnsemble writes its prediction caches to ./cache: every run works in its own folder, so a
    # parallel run (or the leftovers of an aborted one) can never serve or delete this run's cache files.
    work_dir = WORK_DIR
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    os.chdir(work_dir)
    # the scripts write into a staging folder of this run; publish() moves the files to RESULT_DIR under their final names
    staging = work_dir / "out"
    staging.mkdir()
    base.RESULT_DIR = stage1.RESULT_DIR = files.FUSION_DIR = staging  # FUSION_DIR: where the library would save its own jsons
    base.MODEL_DESCRIPTION = ("labelfusion + time series: roberta-large fine-tuned alone, then fusion MLP on frozen embeddings "
                              f"+ gpt-5-nano labels + TimesFM/CrossTSTransformer embedding of {', '.join(TS_SERIES)}")
    base.SETTINGS_EXTRA.update({"ts_series": TS_SERIES, "ts_window_days": args.ts_window or "full history", "ts_dim": args.ts_dim,
                                "ts_pool_only": args.ts_pooling if args.ts_pool_only else False,
                                "ts_finetune": args.ts_finetune, "roberta_params_from": str(args.roberta_params_from)})

    branch = TSBranch(args)
    ts_input = (f"pooled ({args.ts_pooling}) {branch.hidden_size}-dim embeddings, no attention" if args.ts_pool_only
                else f"{branch.output_dim}-dim embedding")
    print(f"time series: {', '.join(TS_SERIES)}, window {args.ts_window or 'full history (series start up to the date)'}"
          f"{'' if not args.ts_window else ' days'}, {ts_input}, TimesFM {'fine-tuned' if args.ts_finetune else 'frozen'}")

    # predict_all_labelFusion.run_pair, with dated data and a fusion MLP that also gets the TS branch.
    plain_build_fusion = base.build_fusion

    ts_out = f"pool{args.ts_pooling}" if args.ts_pool_only else f"d{args.ts_dim}"
    ts_tag = f"ts{args.ts_window or 'full'}{ts_out}{'ft' if args.ts_finetune else ''}"  # settings of the TS branch

    def build_fusion_ts(ml_model, llm_model, fus, name, tag, save):
        # the TS settings are part of the tag, so cache files (and experiment names) of runs with other
        # TS settings never collide
        fusion = plain_build_fusion(ml_model, llm_model, fus, name, f"{tag}_{ts_tag}", save)
        loader, encoders, transformer = branch.fresh()
        fusion.add_ts_model(loader, encoders, transformer, date_column=DATE_COLUMN, window_days=args.ts_window or FULL_HISTORY)
        if args.ts_pool_only:
            transformer.output_dim = transformer.hidden_size  # the fusion MLP reads its TS input size from here
            fusion.ts_pooling = args.ts_pooling
            fusion._run_ts_branch = types.MethodType(run_pooled_ts_branch, fusion)
        return fusion

    base.build_fusion = build_fusion_ts
    base.load = load_dated
    base.split_pair = split_dated

    if args.roberta_params_from:
        reuse = pd.read_csv(args.roberta_params_from).set_index("file")
        plain_tune = stage1.tune

        def tune_or_reuse(train_df, val_df, name, targs):
            if name not in reuse.index:
                print(f"{name} not in {args.roberta_params_from.name}: tuning stage 1")
                return plain_tune(train_df, val_df, name, targs)
            r = reuse.loc[name]
            params = {**stage1.default_params(targs), "learning_rate": float(r["roberta_lr"]),
                      "batch_size": int(r["roberta_batch_size"]), "num_epochs": int(r["roberta_epochs"])}
            print(f"### {name}: reusing roberta-large parameters lr={params['learning_rate']:.2e} "
                  f"batch_size={params['batch_size']} epochs={params['num_epochs']}")
            val_f1 = r.get("roberta_val_f1")
            return params, (None if pd.isna(val_f1) else float(val_f1))

        stage1.tune = tune_or_reuse

    pairs = []
    for test_path in sorted(base.TEST_DIR.glob("lab-manual-*-test-*.xlsx")):
        m = base.SEED_RE.match(test_path.stem)
        if not m or (args.datasets and not any(d in m["dataset"] for d in args.datasets)):
            continue
        train_path = base.paired_train(test_path)
        missing = [p.name for p in (train_path, dated_path(test_path), dated_path(train_path)) if not p.exists()]
        if missing:
            print(f"skip {test_path.stem}: missing {', '.join(missing)}")
            continue
        pairs.append((train_path, test_path))
    if not pairs:
        raise SystemExit("no matching train/test pairs found")
    print(f"{len(pairs)} pairs: {', '.join(p.stem for _, p in pairs)}")

    rows = []
    for train_path, test_path in pairs:
        shutil.rmtree(work_dir / "cache", ignore_errors=True)
        rows.append(base.run_pair(train_path, test_path, args))
        seed_rows = pd.DataFrame(rows)
        seed_rows.to_csv(RESULT_DIR / f"summary_seeds_{SUFFIX}.csv", index=False)
        base.aggregate(seed_rows).to_csv(RESULT_DIR / f"summary_{SUFFIX}.csv", index=False)
        base.write_dataset_json(rows[-1]["dataset"], seed_rows, args)
        publish(staging)

    summary = base.aggregate(pd.DataFrame(rows))
    print("\n=== weighted F1 mean +- std over seeds ===")
    for _, r in summary.iterrows():
        cells = "  ".join(f"{who} {r[f'{who}_f1_weighted_mean']:.3f}+-{r[f'{who}_f1_weighted_std']:.3f}"
                          for who in ("llm", "roberta", "fusion"))
        print(f"{r['dataset']} (n={r['n_seeds']}): {cells}")
    print(f"\nSaved in {RESULT_DIR}: summary_{SUFFIX}.csv, summary_seeds_{SUFFIX}.csv, one "
          f"<dataset>_summary_{SUFFIX}.json per dataset and per pair <test file>_{SUFFIX}.csv/.json (+ trials/)")


if __name__ == "__main__":
    main()
