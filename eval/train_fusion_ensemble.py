"""
Train and evaluate a FusionEnsemble (RoBERTa + OpenAI LLM + timeseries branch)
on the all-commodities article dataset.

The task is multi-label classification: each article is labelled with
which of gold/silver/oil/gas it discusses (an article can mention several).

FusionEnsemble learns a small MLP that fuses RoBERTa's embeddings, the LLM's
zero-shot predictions, and a market-context embedding (TS -> TSEmbedder ->
CrossTSTransformer over the trailing gold/silver/oil/gas price windows), and
is trained/evaluated here on a random subsample of the full dataset (see
SAMPLE_* below) to keep run time and OpenAI API cost low. Bump the sample
sizes once the pipeline is confirmed to work end-to-end.

Note: the fusion MLP is trained only on val_df (see FusionEnsemble.fit),
further split 90/10 internally -- VAL_SIZE therefore matters more than
TRAIN_SIZE for how well the fusion layer itself learns.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from textclassify import RoBERTaClassifier, OpenAIClassifier, FusionEnsemble
from textclassify.core.types import ModelConfig, EnsembleConfig, ModelType
from textclassify.timeseries import TS, TSEmbedder, CrossTSTransformer

# ── Configuration ────────────────────────────────────────────────────────
DATA_PATH = REPO_ROOT / "data" / "articles" / "articles_all_commodities_baseline_seed7_20260722_190748.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
LABEL_COLUMNS = ["gold", "silver", "oil", "gas"]
TEXT_COLUMN = "text"
DATE_COLUMN = "article_date"
RANDOM_STATE = 7

# Small subsample: fast + cheap for verifying the pipeline end-to-end.
# VAL_SIZE bumped from 60 -> 150 so the fusion MLP (trained on val_df, split
# 90/10 internally) gets a less thin ~135/15 train/val split instead of ~54/6.
SAMPLE_SIZE = 550
TRAIN_SIZE = 300
VAL_SIZE = 150
# remaining ~100 rows become the test set

LLM_MODEL = "gpt-4o-mini"
ROBERTA_MODEL = "roberta-base"

TS_DATA_DIR = REPO_ROOT / "data" / "TSs"
TS_SERIES_NAMES = LABEL_COLUMNS  # gold/silver/oil/gas match the label columns exactly
TS_WINDOW_DAYS = 21  # matches the "prices_21d_*" convention already in the article CSV


def load_split_data():
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COLUMN] = df["headline"].fillna("") + " " + df["body"].fillna("")
    df = df[[TEXT_COLUMN, DATE_COLUMN] + LABEL_COLUMNS]

    sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    train_df, rest_df = train_test_split(
        sample_df, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        rest_df, train_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split_df[LABEL_COLUMNS].sum().to_dict()
        print(f"  {name} label counts: {counts}")

    return train_df, val_df, test_df


def build_ml_model() -> RoBERTaClassifier:
    ml_config = ModelConfig(
        model_name=ROBERTA_MODEL,
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            "model_name": ROBERTA_MODEL,
            "max_length": 256,
            "batch_size": 8,
            "num_epochs": 3,
            "learning_rate": 2e-5,
        },
    )
    return RoBERTaClassifier(
        config=ml_config,
        text_column=TEXT_COLUMN,
        label_columns=LABEL_COLUMNS,
        multi_label=True,
        output_dir=str(OUTPUT_DIR),
        experiment_name="fusion_roberta",
    )


def build_llm_model() -> OpenAIClassifier:
    llm_config = ModelConfig(
        model_name=LLM_MODEL,
        model_type=ModelType.LLM,
        parameters={
            "model": LLM_MODEL,
            "temperature": 0.1,
            "max_completion_tokens": 100,
        },
    )
    return OpenAIClassifier(
        config=llm_config,
        text_column=TEXT_COLUMN,
        label_columns=LABEL_COLUMNS,
        multi_label=True,
        output_dir=str(OUTPUT_DIR),
        experiment_name="fusion_openai",
        cache_dir=str(OUTPUT_DIR / "llm_cache"),
    )


def build_ts_branch():
    """Build the TS -> per-commodity TSEmbedders -> CrossTSTransformer trio for add_ts_model().

    Each commodity gets its own TSEmbedder instance (not a shared one), so they can be
    fine-tuned independently during fusion training instead of staying frozen.
    """
    ts_loader = TS(
        data_dir=str(TS_DATA_DIR),
        stock_symbols=TS_SERIES_NAMES,
        date_column="Date",
        price_column="Stock_Price",
    )
    ts_loader.load_all()

    ts_embedders = {name: TSEmbedder(pooling="mean") for name in TS_SERIES_NAMES}
    hidden_size = next(iter(ts_embedders.values())).hidden_size

    ts_transformer = CrossTSTransformer(
        series_names=TS_SERIES_NAMES,
        hidden_size=hidden_size,
        output_dim=256,
    )

    return ts_loader, ts_embedders, ts_transformer


def build_fusion_ensemble(ml_model, llm_model, ts_branch=None) -> FusionEnsemble:
    ensemble_config = EnsembleConfig(
        ensemble_method="fusion",
        models=[ml_model, llm_model],
        parameters={
            "fusion_hidden_dims": [64, 32],
            "ml_lr": 1e-5,
            "fusion_lr": 1e-3,
            "num_epochs": 10,
            "batch_size": 16,
            "classification_type": "multi_label",
            "output_dir": str(OUTPUT_DIR),
            "experiment_name": "fusion_ensemble",
            "auto_save_results": True,
        },
    )

    fusion_ensemble = FusionEnsemble(
        ensemble_config,
        output_dir=str(OUTPUT_DIR),
        experiment_name="fusion_ensemble",
    )
    fusion_ensemble.add_ml_model(ml_model)
    fusion_ensemble.add_llm_model(llm_model)

    if ts_branch is not None:
        ts_loader, ts_embedders, ts_transformer = ts_branch
        fusion_ensemble.add_ts_model(
            ts_loader, ts_embedders, ts_transformer,
            date_column=DATE_COLUMN, window_days=TS_WINDOW_DAYS,
        )

    return fusion_ensemble


def main():
    train_df, val_df, test_df = load_split_data()

    ml_model = build_ml_model()
    llm_model = build_llm_model()
    ts_branch = build_ts_branch()
    fusion_ensemble = build_fusion_ensemble(ml_model, llm_model, ts_branch=ts_branch)

    print("\n=== Training fusion ensemble ===")
    fusion_ensemble.fit(train_df, val_df)

    print("\n=== Evaluating on test set ===")
    result = fusion_ensemble.predict(test_df, train_df=train_df)

    metrics = result.metadata.get("metrics", {}) if result.metadata else {}
    print("\n=== Test metrics ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")

    print("\nSample predictions:")
    for text, pred, true in list(zip(test_df[TEXT_COLUMN], result.predictions, test_df[LABEL_COLUMNS].values.tolist()))[:5]:
        true_labels = [LABEL_COLUMNS[i] for i, v in enumerate(true) if v == 1]
        print(f"  pred={pred}  true={true_labels}  text={text[:80]}...")


if __name__ == "__main__":
    main()
