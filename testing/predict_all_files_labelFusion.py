"""LabelFusion over every train/test file pair: RoBERTa embeddings + LLM predictions,
with the RoBERTa expert and the fusion layer both tuned by Optuna.

For each data/training_data/<name>-train-<seed>.xlsx and its data/test_data/<name>-test-<seed>.xlsx:

  1. The train file is split 80/20 (stratified) into train / validation.
  2. A RoBERTa text expert is fine-tuned on the train split.
  3. The LLM expert is NOT called again: its predictions come from the files written by
     predict_all_files_llm.py (./outputs/predictions/...):
       train + validation rows  <- GPT-5 nano, zero-shot   (train_gpt-5-nano_zero_shot)
       test rows                <- GPT-5 nano, 5-shot      (test_gpt-5-nano_5_shot)
  4. textclassify's FusionEnsemble (FusionWrapper + FusionMLP) concatenates the RoBERTa
     [CLS] embedding (768) with the LLM's class vector and trains the fusion MLP on the
     validation split; RoBERTa and the LLM stay frozen at that point.
  5. The fused model predicts the test file.

Optuna (--trials, default 10) tunes both parts JOINTLY. One trial =
  RoBERTa: learning rate, epochs (1-4), batch size, weight decay, max_length
  fusion MLP: hidden layers, learning rate, epochs, batch size
and consists of a full RoBERTa fine-tune on the train split, then the fusion MLP trained on
half of the validation split, scored (macro-F1) on the other half. Test files are never used
to pick parameters. The default configuration is always trial 0, so the tuned result cannot
be worse than the default on that score. The best parameters are then used for the real run
on the pair: RoBERTa on train, fusion MLP on the full validation split, prediction on test.

--tune-scope decides how many studies there are (each trial is a CPU fine-tune, so this is the
main cost knob):
  group   (default) one study per dataset (mm, pc, sp, combine, ...) on its seed-5768 pair;
                    the best parameters are reused for the other seeds of that dataset.
  shared            one study (--tune-file) whose best parameters are used for every pair.
  pair              a separate study for every pair (most expensive).
Caveat for group/shared: the seeds are different resamples of the same data, so a sentence in
another seed's test file may have been in the tuning pair's validation half. Only
hyperparameters are shared, but the selection is not perfectly independent of those test files.

Note on "logits": the LLM only returns a hard class choice, so the fusion input is a one-hot
vector (dovish / hawkish / neutral), not logits.

Writes ./outputs/fusion/<test file name>.csv and .json (sentence, label, true, llm_pred,
roberta_pred, fusion_pred, plus *_default_pred columns with --baseline),
./outputs/fusion/summary.csv (accuracy / macro-F1 of LLM alone, RoBERTa alone, fusion, and the
chosen parameters) and ./outputs/fusion/optuna/<study>_trials.csv (every trial). Studies live in
./outputs/fusion/optuna.db and resume where they stopped; finished pairs are skipped unless
--force. Fine-tuned RoBERTa models are never kept (they are ~500 MB each): the classifier is
pointed at a cache outside the repo (LABELFUSION_MODEL_CACHE, default ~/.cache/labelfusion) and
that copy is deleted right after training.

    python testing/predict_all_files_labelFusion.py                            # everything
    python testing/predict_all_files_labelFusion.py --files mm-test-5768       # matching pairs only
    python testing/predict_all_files_labelFusion.py --tune-scope shared --tune-file lab-manual-mm-test-5768
    python testing/predict_all_files_labelFusion.py --trials 0                 # no tuning (defaults)
    python testing/predict_all_files_labelFusion.py --files pc-test-5768 --limit-train 60 --trials 3 --epochs 1   # smoke test

CPU cost (no GPU here): about 3-6 s per step of 16 sentences, i.e. roughly 5-10 minutes per epoch
for the larger files. A trial is therefore 5-40 minutes on a large pair. See --tune-minutes to cap
each study.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import os
import re
import shutil
import sys
import time
from pathlib import Path

import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

TESTING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING_DIR))

from test_textclassify_llm import (  # noqa: E402  (also loads .env)
    LABEL_COLUMNS, LABEL_MAP, RANDOM_STATE, REPO_ROOT, TEXT_COLUMN, _add_onehot_labels,
)
from textclassify import FusionEnsemble, RoBERTaClassifier  # noqa: E402
from textclassify.core.types import (  # noqa: E402
    ClassificationResult, ClassificationType, EnsembleConfig, ModelConfig, ModelType,
)

TRAIN_DIR = REPO_ROOT / "data" / "training_data"
TEST_DIR = REPO_ROOT / "data" / "test_data"
OUT_DIR = REPO_ROOT / "outputs"
LLM_TRAIN_DIR = OUT_DIR / "predictions" / "train_gpt-5-nano_zero_shot"
LLM_TEST_DIR = OUT_DIR / "predictions" / "test_gpt-5-nano_5_shot"
FUSION_DIR = OUT_DIR / "fusion"
# Fine-tuned models are written here (outside the repo / ownCloud) and deleted right away.
MODEL_CACHE = Path(os.getenv("LABELFUSION_MODEL_CACHE", Path.home() / ".cache" / "labelfusion"))

ROBERTA_MODEL = os.getenv("TEXTCLASSIFY_ROBERTA_MODEL", "roberta-base")
HIDDEN_DIMS = {"32": [32], "64-32": [64, 32], "128-64": [128, 64], "256-128-64": [256, 128, 64]}


class PrecomputedLLM:
    """Stands in for the LLM expert inside FusionEnsemble: serves predictions that were
    already made (predict_all_files_llm.py) instead of calling an API. FusionEnsemble.fit()
    calls llm_model.fit(train_df, val_df) and takes the train/val predictions from its result."""

    def __init__(self, pred_by_text: dict[str, str]):
        self.pred_by_text = pred_by_text
        self.classes_ = LABEL_COLUMNS
        self.label_columns = LABEL_COLUMNS
        self.text_column = TEXT_COLUMN
        self.multi_label = False
        self.is_trained = True
        self.results_manager = None

    def lookup(self, df: pd.DataFrame) -> list[str]:
        return [self.pred_by_text[t] for t in df[TEXT_COLUMN]]

    def predict_texts(self, texts, true_labels=None, **_) -> ClassificationResult:
        # FusionEnsemble.predict calls this for LLM metrics, then overrides the predictions.
        return ClassificationResult(
            predictions=[self.pred_by_text.get(t, LABEL_COLUMNS[-1]) for t in texts], model_name="precomputed_llm"
        )

    def fit(self, train_df=None, val_df=None, **_) -> dict:
        return {
            split: ClassificationResult(predictions=self.lookup(df), model_name="precomputed_llm")
            for split, df in (("train", train_df), ("val", val_df))
            if df is not None
        }


class EmbeddingMemo:
    """Caches the trained RoBERTa's prediction and embedding per sentence. FusionEnsemble asks
    the ML model for predictions on every fit()/predict(); with the memo the transformer runs
    once per sentence, so re-fitting the fusion MLP is cheap."""

    def __init__(self, ml_model):
        self.cache: dict[str, tuple] = {}
        self._predict = ml_model.predict_without_saving
        ml_model.predict_without_saving = self.predict
        ml_model.predict = self.predict  # FusionEnsemble.predict() calls ml_model.predict(test_df)

    def predict(self, df: pd.DataFrame, mode: str = "test") -> ClassificationResult:
        missing = df[~df[TEXT_COLUMN].isin(self.cache)].drop_duplicates(TEXT_COLUMN).reset_index(drop=True)
        if len(missing):
            res = self._predict(missing, mode="test")
            for text, pred, emb in zip(missing[TEXT_COLUMN], res.predictions, res.embeddings):
                self.cache[text] = (pred, emb)
        rows = [self.cache[t] for t in df[TEXT_COLUMN]]
        return ClassificationResult(
            predictions=[p for p, _ in rows],
            embeddings=[e for _, e in rows],
            model_name="roberta_memo",
            classification_type=ClassificationType.MULTI_CLASS,
        )


def load(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    return _add_onehot_labels(df[[TEXT_COLUMN, "label"]].copy()).reset_index(drop=True)


def llm_predictions(csv_path: Path) -> dict[str, str]:
    """sentence -> class name from a predict_all_files_llm.py output CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} missing -- run predict_all_files_llm.py first")
    df = pd.read_csv(csv_path)
    return dict(zip(df[TEXT_COLUMN], df["pred"]))


def paired_train(test_path: Path) -> Path:
    return TRAIN_DIR / test_path.name.replace("-test-", "-train-")


def split_pair(train_path: Path, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train file -> (train, validation), 80/20 stratified."""
    train_full = load(train_path)
    if args.limit_train:
        train_full = train_full.sample(n=args.limit_train, random_state=RANDOM_STATE).reset_index(drop=True)
    train_df, val_df = train_test_split(
        train_full, train_size=0.8, random_state=RANDOM_STATE, stratify=train_full["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def default_params(args) -> tuple[dict, dict]:
    roberta = {"learning_rate": 2e-5, "num_epochs": args.epochs, "batch_size": 16, "weight_decay": 0.01, "max_length": 128}
    fusion = {"fusion_hidden_dims": [64, 32], "fusion_lr": 1e-3, "num_epochs": args.fusion_epochs, "batch_size": 16}
    return roberta, fusion


def build_roberta(rob: dict, name: str, tag: str, save: bool) -> RoBERTaClassifier:
    config = ModelConfig(
        model_name=ROBERTA_MODEL,
        model_type=ModelType.TRADITIONAL_ML,
        parameters={"model_name": ROBERTA_MODEL, **rob},
    )
    return RoBERTaClassifier(
        config=config,
        text_column=TEXT_COLUMN,
        label_columns=LABEL_COLUMNS,
        multi_label=False,
        auto_save_results=save,
        output_dir=str(FUSION_DIR),
        experiment_name=f"roberta_{name}_{tag}",
        cache_dir=str(MODEL_CACHE),
    )


def delete_saved_model(train_df: pd.DataFrame) -> None:
    """RoBERTaClassifier.fit() always saves the model to <cache_dir>/roberta_<hash of train texts>."""
    text_hash = hashlib.md5(pd.util.hash_pandas_object(train_df[TEXT_COLUMN], index=False).values).hexdigest()[:8]
    shutil.rmtree(MODEL_CACHE / f"roberta_{text_hash}", ignore_errors=True)


def build_fusion(ml_model, llm_model, fus: dict, name: str, tag: str, save: bool) -> FusionEnsemble:
    """FusionEnsemble with the given fusion-MLP params. `tag` makes the ensemble's test-prediction
    cache path unique, so predictions of one parameter set are never served for another."""
    config = EnsembleConfig(
        ensemble_method="fusion",
        models=[ml_model, llm_model],
        parameters={
            **fus,
            "ml_lr": 1e-5,
            "classification_type": "multi_class",
            "output_dir": str(FUSION_DIR),
            "experiment_name": f"fusion_{name}_{tag}",
            "auto_save_results": save,
            "test_llm_cache_path": f"cache/predictions/{name}_{tag}_llm",
        },
    )
    fusion = FusionEnsemble(config, output_dir=str(FUSION_DIR), experiment_name=f"fusion_{name}_{tag}")
    fusion.add_ml_model(ml_model)
    fusion.add_llm_model(llm_model)
    return fusion


def quiet(fn, *a, **kw):
    """Run fn without the library's (very chatty) progress prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def fit_predict(rob, fus, train_df, fit_df, pred_df, pred_llm, llm, name, tag, save, verbose):
    """RoBERTa on train_df, fusion MLP on fit_df, then predict pred_df.
    Returns (fusion predictions, RoBERTa-alone predictions)."""
    run = (lambda fn, *a, **kw: fn(*a, **kw)) if verbose else quiet
    ml = build_roberta(rob, name, tag, save)
    run(ml.fit, train_df, fit_df)
    delete_saved_model(train_df)
    EmbeddingMemo(ml)
    fusion = build_fusion(ml, llm, fus, name, tag, save)
    run(fusion.fit, train_df, fit_df)
    fusion_preds = run(fusion.predict, pred_df, train_df=train_df, test_llm_predictions=pred_llm).predictions
    roberta_preds = ml.predict_without_saving(pred_df).predictions
    del fusion, ml
    gc.collect()
    return fusion_preds, roberta_preds


def scores(true: list[str], pred: list[str]) -> dict:
    return {"accuracy": accuracy_score(true, pred), "f1_macro": f1_score(true, pred, average="macro")}


def study_key(test_stem: str, scope: str, args) -> str:
    if scope == "pair":
        return test_stem
    if scope == "shared":
        return "shared"
    return re.sub(r"-\d+$", "", test_stem)  # group: dataset without the seed


def tuning_pair(test_stem: str, scope: str, args) -> tuple[Path, Path]:
    """(train file, test file) the study for this pair is run on."""
    if scope == "pair":
        stem = test_stem
    elif scope == "shared":
        stem = args.tune_file
    else:
        group = re.sub(r"-\d+$", "", test_stem)
        stem = f"{group}-{args.tune_seed}"
    test_path = TEST_DIR / f"{stem}.xlsx"
    return paired_train(test_path), test_path


def suggest(trial: optuna.Trial) -> tuple[dict, dict]:
    rob = {
        "learning_rate": trial.suggest_float("roberta_lr", 1e-5, 5e-5, log=True),
        "num_epochs": trial.suggest_int("roberta_epochs", 1, 4),
        "batch_size": trial.suggest_categorical("roberta_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_categorical("roberta_weight_decay", [0.0, 0.01, 0.1]),
        "max_length": trial.suggest_categorical("roberta_max_length", [64, 96, 128]),
    }
    fus = {
        "fusion_hidden_dims": HIDDEN_DIMS[trial.suggest_categorical("hidden_dims", list(HIDDEN_DIMS))],
        "fusion_lr": trial.suggest_float("fusion_lr", 1e-4, 1e-2, log=True),
        "num_epochs": trial.suggest_int("fusion_epochs", 10, 80, step=10),
        "batch_size": trial.suggest_categorical("fusion_batch_size", [8, 16, 32]),
    }
    return rob, fus


def params_from(best: dict) -> tuple[dict, dict]:
    rob = {
        "learning_rate": best["roberta_lr"], "num_epochs": best["roberta_epochs"],
        "batch_size": best["roberta_batch_size"], "weight_decay": best["roberta_weight_decay"],
        "max_length": best["roberta_max_length"],
    }
    fus = {
        "fusion_hidden_dims": HIDDEN_DIMS[best["hidden_dims"]], "fusion_lr": best["fusion_lr"],
        "num_epochs": best["fusion_epochs"], "batch_size": best["fusion_batch_size"],
    }
    return rob, fus


def tune(key: str, train_path: Path, test_path: Path, args) -> tuple[dict, dict, optuna.Study]:
    """Joint Optuna study (RoBERTa + fusion MLP) on one pair. The validation split is halved:
    the fusion MLP trains on one half and every trial is scored (macro-F1) on the other."""
    name = test_path.stem
    train_df, val_df = split_pair(train_path, args)
    try:
        fit_df, eval_df = train_test_split(val_df, train_size=0.5, random_state=RANDOM_STATE, stratify=val_df["label"])
    except ValueError:  # a class too small to stratify
        fit_df, eval_df = train_test_split(val_df, train_size=0.5, random_state=RANDOM_STATE)
    fit_df, eval_df = fit_df.reset_index(drop=True), eval_df.reset_index(drop=True)
    llm = PrecomputedLLM(llm_predictions(LLM_TRAIN_DIR / f"{train_path.stem}.csv"))
    eval_llm, eval_true = llm.lookup(eval_df), eval_df["label"].map(LABEL_MAP).tolist()
    print(f"\n### Optuna study '{key}' on {name}: train {len(train_df)} / fit {len(fit_df)} / eval {len(eval_df)}")

    def objective(trial: optuna.Trial) -> float:
        rob, fus = suggest(trial)
        preds, _ = fit_predict(rob, fus, train_df, fit_df, eval_df, eval_llm, llm, name,
                               f"trial{trial.number}", save=False, verbose=False)
        return f1_score(eval_true, preds, average="macro")

    study = optuna.create_study(
        study_name=f"{key}_e{args.epochs}_n{args.limit_train or 'all'}",
        storage=f"sqlite:///{FUSION_DIR / 'optuna.db'}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        load_if_exists=True,
    )
    if not study.trials:  # the default configuration is always one of the candidates
        rob, fus = default_params(args)
        study.enqueue_trial({
            "roberta_lr": rob["learning_rate"], "roberta_epochs": min(max(rob["num_epochs"], 1), 4),
            "roberta_batch_size": rob["batch_size"], "roberta_weight_decay": rob["weight_decay"],
            "roberta_max_length": rob["max_length"], "hidden_dims": "64-32", "fusion_lr": fus["fusion_lr"],
            "fusion_epochs": fus["num_epochs"], "fusion_batch_size": fus["batch_size"],
        })
    # Count finished trials only: the enqueued default trial is still WAITING and must run.
    remaining = args.trials - sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, timeout=args.tune_minutes * 60 if args.tune_minutes else None)
    (FUSION_DIR / "optuna").mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(FUSION_DIR / "optuna" / f"{key}_trials.csv", index=False)
    rob, fus = params_from(study.best_trial.params)
    print(f"### best macro-F1 {study.best_value:.3f} after {len(study.trials)} trials: {rob} {fus}")
    return rob, fus, study


def run_pair(train_path: Path, test_path: Path, rob: dict, fus: dict, extra: dict, args) -> dict:
    name = test_path.stem
    t0 = time.time()
    train_df, val_df = split_pair(train_path, args)
    test_df = load(test_path)
    print(f"\n=== {name}: train {len(train_df)} / val {len(val_df)} / test {len(test_df)} ===")

    llm = PrecomputedLLM(llm_predictions(LLM_TRAIN_DIR / f"{train_path.stem}.csv"))
    llm_test = llm_predictions(LLM_TEST_DIR / f"{test_path.stem}.csv")
    llm_test_preds = [llm_test[t] for t in test_df[TEXT_COLUMN]]

    fusion_preds, roberta_preds = fit_predict(
        rob, fus, train_df, val_df, test_df, llm_test_preds, llm, name, "final", save=True, verbose=True
    )
    out = pd.DataFrame({
        TEXT_COLUMN: test_df[TEXT_COLUMN],
        "label": test_df["label"],
        "true": test_df["label"].map(LABEL_MAP),
        "llm_pred": llm_test_preds,
        "roberta_pred": roberta_preds,
        "fusion_pred": fusion_preds,
    })
    if args.baseline and args.trials > 0:  # default parameters, for comparison with the tuned ones
        d_rob, d_fus = default_params(args)
        d_fusion, d_roberta = fit_predict(
            d_rob, d_fus, train_df, val_df, test_df, llm_test_preds, llm, name, "default", save=False, verbose=False
        )
        out["roberta_default_pred"], out["fusion_default_pred"] = d_roberta, d_fusion
    shutil.rmtree(FUSION_DIR / "cache" / "predictions", ignore_errors=True)  # per-run prediction cache files
    out.to_csv(FUSION_DIR / f"{name}.csv", index=False)
    out.to_json(FUSION_DIR / f"{name}.json", orient="records", force_ascii=False, indent=2)

    row = {"file": name, "rows": len(out), "seconds": round(time.time() - t0)}
    for who in ("llm", "roberta", "fusion", "roberta_default", "fusion_default"):
        if f"{who}_pred" in out:
            row.update({f"{who}_{m}": v for m, v in scores(out["true"].tolist(), out[f"{who}_pred"].tolist()).items()})
    row.update(extra, roberta_params=str(rob), fusion_params=str(fus))
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if k not in ("roberta_params", "fusion_params")})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", help="only test files whose name contains one of these")
    parser.add_argument("--trials", type=int, default=10, help="Optuna trials per study (0 = no tuning, defaults)")
    parser.add_argument("--tune-scope", choices=["group", "shared", "pair"], default="group")
    parser.add_argument("--tune-file", default="lab-manual-mm-test-5768", help="test file stem tuned on with --tune-scope shared")
    parser.add_argument("--tune-seed", default="5768", help="seed of the pair a group is tuned on")
    parser.add_argument("--tune-minutes", type=float, help="time cap per study (finished trials are kept)")
    parser.add_argument("--baseline", action="store_true", help="also run the default parameters on each pair")
    parser.add_argument("--epochs", type=int, default=3, help="RoBERTa epochs of the default configuration")
    parser.add_argument("--fusion-epochs", type=int, default=30, help="fusion MLP epochs of the default configuration")
    parser.add_argument("--limit-train", type=int, help="subsample the train file to N rows (smoke test)")
    parser.add_argument("--force", action="store_true", help="redo pairs that already have results")
    args = parser.parse_args()

    FUSION_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(FUSION_DIR)  # FusionEnsemble writes its prediction caches to ./cache

    best_by_key: dict[str, tuple] = {}
    rows = []
    for test_path in sorted(TEST_DIR.glob("lab-manual-*-test-*.xlsx")):
        train_path = paired_train(test_path)
        if not train_path.exists() or (args.files and not any(f in test_path.stem for f in args.files)):
            continue
        if (FUSION_DIR / f"{test_path.stem}.csv").exists() and not args.force:
            print(f"skip {test_path.stem}: result exists (use --force to redo)")
            continue

        extra = {}
        if args.trials > 0:
            key = study_key(test_path.stem, args.tune_scope, args)
            if key not in best_by_key:
                t_train, t_test = tuning_pair(test_path.stem, args.tune_scope, args)
                *params, study = tune(key, t_train, t_test, args)
                best_by_key[key] = (*params, study.best_value)
            rob, fus, best_f1 = best_by_key[key]
            extra = {"study": key, "optuna_best_f1": best_f1}
        else:
            rob, fus = default_params(args)
        rows.append(run_pair(train_path, test_path, rob, fus, extra, args))

        # Rewrite the summary after every pair so a stopped run keeps its progress.
        summary_path = FUSION_DIR / "summary.csv"
        new = pd.DataFrame(rows)
        if summary_path.exists():
            old = pd.read_csv(summary_path)
            new = pd.concat([old[~old["file"].isin(new["file"])], new], ignore_index=True)
        new.sort_values("file").to_csv(summary_path, index=False)

    if rows:
        print(f"\nSaved: {FUSION_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
