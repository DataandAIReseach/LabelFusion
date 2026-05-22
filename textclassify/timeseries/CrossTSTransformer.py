import torch
import torch.nn as nn
from typing import Dict

from .TSEmbeddingFuser import TSEmbeddingFuser


class CrossTSTransformer(nn.Module):
    """
    Applies cross-series attention on top of TSEmbeddingFuser.

    NEW ARCHITECTURE:
        The fuser now receives pre-computed embeddings instead of raw series.
        This transformer works with the embeddings dict format.

    Each stock embedding is treated as a token. Multi-head attention
    lets every series attend to all others, capturing market interactions.

    Architecture:
        N × (B, 1280)          per-series embeddings from TSEmbeddingFuser
            ↓ stack as sequence
        (B, N, 1280)           series as tokens
            ↓ TransformerEncoder (cross-series attention)
        (B, N, 1280)           interaction-aware embeddings
            ↓ mean pool over series
        (B, 1280)
            ↓ Linear projection
        (B, output_dim)         final representation

    Args:
        series_names: List of series names (must match fuser)
        hidden_size: Embedding dimension (default: 1280)
        output_dim: Final output dimensionality (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of TransformerEncoder layers (default: 2)
        dropout: Dropout in attention and feedforward (default: 0.1)
        feedforward_dim: Inner dim of transformer FFN (default: 2048)
    """

    def __init__(
        self,
        series_names: list[str],
        hidden_size: int = 1280,
        output_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        feedforward_dim: int = 2048,
    ):
        super().__init__()

        self.series_names = list(series_names)
        self.hidden_size = hidden_size
        self.n_series = len(series_names)
        self.output_dim = output_dim

        # Learnable positional embedding — one per series slot
        self.pos_embedding = nn.Embedding(self.n_series, self.hidden_size)

        # Transformer encoder — operates over the series dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,       # (B, seq, dim)
            norm_first=True,        # pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Project pooled representation to output_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, output_dim),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        embeddings_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings_dict: dict {series_name: (B, hidden_size)}
                            Pre-computed embeddings from TSEmbedder

        Returns:
            dict with keys:
                "output"        – (B, output_dim)  final compressed representation
                "pooled"        – (B, hidden_size) mean-pooled before projection
                "attended"      – (B, n_series, hidden_size) post-attention tokens
                "per_series"    – {name: (B, hidden_size)} input embeddings
        """
        # 1. Stack embeddings into sequence: (B, n_series, hidden_size)
        tokens = torch.stack(
            [embeddings_dict[name] for name in self.series_names],
            dim=1,
        )  # (B, N, 1280)

        # 2. Add positional embeddings (one per series slot)
        positions = torch.arange(self.n_series, device=tokens.device)
        tokens = tokens + self.pos_embedding(positions).unsqueeze(0)  # (B, N, 1280)

        # 3. Cross-series attention
        attended = self.transformer(tokens)      # (B, N, 1280)

        # 4. Mean pool over series dimension → single vector
        pooled = attended.mean(dim=1)            # (B, 1280)

        # 5. Project to output_dim
        output = self.projection(pooled)         # (B, output_dim)

        return {
            "output": output,
            "pooled": pooled,
            "attended": attended,
            "per_series": embeddings_dict.copy(),
        }


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------