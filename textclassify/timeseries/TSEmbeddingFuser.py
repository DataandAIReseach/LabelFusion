import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .TSEmbedder import TSEmbedder


class TSEmbeddingFuser(nn.Module):
    """
    Embeds multiple named time series independently with TSEmbedder
    and concatenates their embeddings into one big flat vector per sample.

    Args:
        series_names (list[str]): Ordered list of channel/series names.
        embedder (TSEmbedder | None): A pre-built embedder to reuse.
            If None, a new one is created from the remaining kwargs.
        pretrained_model_name_or_path (str): Passed to TSEmbedder if
            no embedder is provided.
        pooling (str): "mean" or "last". "none" is not supported here
            because we need a fixed-size vector per series to concatenate.
        device_map: Passed to TSEmbedder if no embedder is provided.

    Shape:
        Input : dict {series_name: list_of_1D_tensors}  length B each
        Output: torch.Tensor (B, len(series_names) * hidden_size)
    """

    def __init__(
        self,
        series_names: List[str],
        embedder: Optional[TSEmbedder] = None,
        pretrained_model_name_or_path: str = "google/timesfm-2.5-200m-transformers",
        pooling: str = "mean",
        device_map=None,
    ):
        super().__init__()

        if not series_names:
            raise ValueError("series_names must not be empty")
        if pooling == "none":
            raise ValueError(
                "pooling='none' produces variable-size tensors and cannot be stacked. "
                "Use 'mean' or 'last'."
            )

        self.series_names = list(series_names)

        self.embedder = embedder or TSEmbedder(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pooling=pooling,
            device_map=device_map,
        )

        self.hidden_size = self.embedder.hidden_size
        self.stacked_size = len(self.series_names) * self.hidden_size

    @property
    def device(self) -> torch.device:
        return self.embedder.device

    def _validate_input(self, series_dict: Dict[str, list], batch_size: int):
        missing = [n for n in self.series_names if n not in series_dict]
        if missing:
            raise KeyError(f"Missing series in input dict: {missing}")
        for name in self.series_names:
            if len(series_dict[name]) != batch_size:
                raise ValueError(
                    f"Series '{name}' has {len(series_dict[name])} samples "
                    f"but expected {batch_size}."
                )

    def forward(
        self,
        series_dict: Dict[str, list],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            series_dict: dict mapping each series name to a list of B
                         1-D array-like time series (numpy / list / tensor).

        Returns:
            dict with keys:
                "stacked"      – (B, N * hidden_size)  the big flat vector
                "per_series"   – dict {name: (B, hidden_size)}
                "stacked_size" – int, total dimension of the stacked vector
        """
        batch_size = len(series_dict[self.series_names[0]])
        self._validate_input(series_dict, batch_size)

        per_series: Dict[str, torch.Tensor] = {}
        for name in self.series_names:
            per_series[name] = self.embedder(series_dict[name])["embeddings"]

        stacked = torch.cat(
            [per_series[name] for name in self.series_names],
            dim=1,
        )

        return {
            "stacked": stacked,
            "per_series": per_series,
            "stacked_size": self.stacked_size,
        }