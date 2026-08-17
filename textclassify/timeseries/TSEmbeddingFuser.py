"""
TSEmbeddingFuser - Receives embeddings directly and fuses them.

CLEAN ARCHITECTURE:
    TS("AAPL") → series → TSEmbedder → embedding_aapl (B, 1280)
    TS("MSFT") → series → TSEmbedder → embedding_msft (B, 1280)
                                            ↓
                        TSEmbeddingFuser({AAPL: emb, MSFT: emb})
                                            ↓
                                    fused_embedding (B, 2560)

The Fuser does NOT create or call TSEmbedder internally.
It only concatenates/fuses pre-computed embeddings.
"""

import torch
import torch.nn as nn
from typing import Dict, List


class TSEmbeddingFuser(nn.Module):
    """
    TSEmbeddingFuser that receives embeddings directly (no internal TSEmbedder).
    
    This version does NOT create TSEmbedder internally.
    It only concatenates embeddings that have already been computed.
    
    Args:
        series_names: List of series names (e.g., ["AAPL", "MSFT"])
        hidden_size: Dimension of each embedding (default: 1280 for TimesFM)
    
    Usage:
        # Step 1: Embed each series separately
        embedder = TSEmbedder(pooling="mean")
        
        embedding_aapl = embedder([series_aapl])["embeddings"]  # (1, 1280)
        embedding_msft = embedder([series_msft])["embeddings"]  # (1, 1280)
        
        # Step 2: Fuse embeddings
        fuser = TSEmbeddingFuser(series_names=["AAPL", "MSFT"])
        
        embeddings_dict = {
            "AAPL": embedding_aapl,
            "MSFT": embedding_msft
        }
        
        output = fuser(embeddings_dict)
        # output["stacked"]: (1, 2560)
    """
    
    def __init__(
        self,
        series_names: List[str],
        hidden_size: int = 1280,
    ):
        super().__init__()
        
        if not series_names:
            raise ValueError("series_names must not be empty")
        
        self.series_names = list(series_names)
        self.hidden_size = hidden_size
        self.stacked_size = len(self.series_names) * self.hidden_size
    
    @property
    def device(self) -> torch.device:
        # Default to CPU since we don't own any parameters
        return torch.device('cpu')
    
    def _validate_input(self, embeddings_dict: Dict[str, torch.Tensor], batch_size: int):
        """Validate embeddings_dict format."""
        missing = [n for n in self.series_names if n not in embeddings_dict]
        if missing:
            raise KeyError(f"Missing series in embeddings_dict: {missing}")
        
        for name in self.series_names:
            emb = embeddings_dict[name]
            if not isinstance(emb, torch.Tensor):
                raise TypeError(f"Embedding for '{name}' must be a torch.Tensor, got {type(emb)}")
            
            if emb.shape[0] != batch_size:
                raise ValueError(
                    f"Embedding for '{name}' has batch size {emb.shape[0]} "
                    f"but expected {batch_size}"
                )
            
            if emb.shape[1] != self.hidden_size:
                raise ValueError(
                    f"Embedding for '{name}' has hidden size {emb.shape[1]} "
                    f"but expected {self.hidden_size}"
                )
    
    def forward(
        self,
        embeddings_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse pre-computed embeddings.
        
        Args:
            embeddings_dict: Dict mapping series name to embedding tensor
                            e.g., {"AAPL": Tensor(B, 1280), "MSFT": Tensor(B, 1280)}
        
        Returns:
            dict with keys:
                "stacked"      – (B, N * hidden_size) concatenated embeddings
                "per_series"   – dict {name: (B, hidden_size)} original embeddings
                "stacked_size" – int, total dimension
        """
        # Get batch size from first embedding
        batch_size = embeddings_dict[self.series_names[0]].shape[0]
        
        # Validate all embeddings
        self._validate_input(embeddings_dict, batch_size)
        
        # Concatenate in the order of series_names
        stacked = torch.cat(
            [embeddings_dict[name] for name in self.series_names],
            dim=1,
        )
        
        return {
            "stacked": stacked,
            "per_series": embeddings_dict.copy(),
            "stacked_size": self.stacked_size,
        }


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------