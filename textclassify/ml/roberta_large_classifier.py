"""RoBERTa-large text classifier."""

from typing import Any, Dict, List, Optional

from .roberta_classifier import RoBERTaClassifier

# Defaults for roberta-large; anything set in config.parameters takes precedence. The learning
# rate is lower than RoBERTaClassifier's 2e-5 because roberta-large fine-tuning diverges more
# easily (the RoBERTa paper used 1e-5 for the large model on GLUE).
ROBERTA_LARGE_DEFAULTS: Dict[str, Any] = {
    "model_name": "roberta-large",
    "learning_rate": 1e-5,
}


class RoBERTaLargeClassifier(RoBERTaClassifier):
    """RoBERTaClassifier with roberta-large (24 layers, 1024-dim [CLS] embedding, ~355M parameters).

    Everything else -- training, prediction, embeddings for FusionEnsemble, saving/loading -- is
    inherited. FusionEnsemble sizes its fusion MLP from `embedding_dim`, which reads the hidden
    size from the loaded model, so the 1024-dim embeddings need no extra configuration.

    Another large checkpoint (e.g. "FacebookAI/roberta-large" or a domain-adapted one) can be
    used by setting config.parameters["model_name"].
    """

    def __init__(
        self,
        config,
        text_column: str = 'text',
        label_columns: Optional[List[str]] = None,
        multi_label: bool = False,
        enable_validation: bool = True,
        auto_save_path: Optional[str] = None,
        auto_save_results: bool = True,
        output_dir: str = "outputs",
        experiment_name: Optional[str] = None,
        cache_dir: str = "cache"
    ):
        config.parameters = {**ROBERTA_LARGE_DEFAULTS, **(config.parameters or {})}
        super().__init__(
            config,
            text_column=text_column,
            label_columns=label_columns,
            multi_label=multi_label,
            enable_validation=enable_validation,
            auto_save_path=auto_save_path,
            auto_save_results=auto_save_results,
            output_dir=output_dir,
            experiment_name=experiment_name,
            cache_dir=cache_dir,
        )
