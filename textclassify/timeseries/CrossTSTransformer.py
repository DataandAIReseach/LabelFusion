import torch
import torch.nn as nn
from typing import Dict

from TSEmbeddingFuser import TSEmbeddingFuser


class CrossTSTransformer(nn.Module):
    """
    Applies cross-series attention on top of TSEmbeddingFuser.

    Each commodity embedding is treated as a token. Multi-head attention
    lets every series attend to all others, capturing market interactions
    (e.g. gold ↔ silver, crude ↔ fuel). A final linear projection
    compresses the result into a single fixed-size representation.

    Architecture:
        59 × (B, 1280)          per-series embeddings from TSEmbeddingFuser
            ↓ stack as sequence
        (B, 59, 1280)           series as tokens
            ↓ TransformerEncoder (cross-series attention)
        (B, 59, 1280)           interaction-aware embeddings
            ↓ mean pool over series
        (B, 1280)
            ↓ Linear projection
        (B, output_dim)         final representation

    Args:
        stacked_embedder (TSEmbeddingFuser): Pre-built embedder.
        output_dim (int): Final output dimensionality. Default 256.
        num_heads (int): Number of attention heads. Must divide hidden_size.
            Default 8 (1280 / 8 = 160 per head).
        num_layers (int): Number of TransformerEncoder layers. Default 2.
        dropout (float): Dropout in attention and feedforward. Default 0.1.
        feedforward_dim (int): Inner dim of the transformer FFN. Default 2048.
    """

    def __init__(
        self,
        stacked_embedder: TSEmbeddingFuser,
        output_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        feedforward_dim: int = 2048,
    ):
        super().__init__()

        self.stacked_embedder = stacked_embedder
        self.hidden_size = stacked_embedder.hidden_size      # 1280
        self.n_series = len(stacked_embedder.series_names)
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
        return self.stacked_embedder.device

    def forward(
        self,
        series_dict: Dict[str, list],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            series_dict: dict {series_name: list of B 1-D time series}.
                         Same format as TSEmbeddingFuser.forward().

        Returns:
            dict with keys:
                "output"        – (B, output_dim)  final compressed representation
                "pooled"        – (B, hidden_size) mean-pooled before projection
                "attended"      – (B, n_series, hidden_size) post-attention tokens
                "per_series"    – {name: (B, hidden_size)} raw embeddings
        """
        # 1. Get per-series embeddings from the stacked embedder
        stacked_out = self.stacked_embedder(series_dict)
        per_series = stacked_out["per_series"]   # {name: (B, 1280)}

        # 2. Stack into sequence: (B, n_series, hidden_size)
        tokens = torch.stack(
            [per_series[name] for name in self.stacked_embedder.series_names],
            dim=1,
        )  # (B, N, 1280)

        # 3. Add positional embeddings (one per series slot)
        positions = torch.arange(self.n_series, device=tokens.device)
        tokens = tokens + self.pos_embedding(positions).unsqueeze(0)  # (B, N, 1280)

        # 4. Cross-series attention
        attended = self.transformer(tokens)      # (B, N, 1280)

        # 5. Mean pool over series dimension → single vector
        pooled = attended.mean(dim=1)            # (B, 1280)

        # 6. Project to output_dim
        output = self.projection(pooled)         # (B, output_dim)

        return {
            "output": output,
            "pooled": pooled,
            "attended": attended,
            "per_series": per_series,
        }