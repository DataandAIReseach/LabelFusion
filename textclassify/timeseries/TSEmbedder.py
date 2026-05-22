import numpy as np
import torch
import torch.nn as nn

# Robust import with fallback
try:
    from transformers import TimesFm2_5Config, TimesFm2_5Model
except ImportError:
    from transformers import AutoModel as TimesFm2_5Model
    from transformers import AutoConfig as TimesFm2_5Config
    print("Warning: TimesFm2_5Model not found, using AutoModel fallback")


class TSEmbedder(nn.Module):
    """
    Wraps TimesFm2_5Model to extract patch-level or sequence-level embeddings
    instead of producing forecasts.

    Args:
        pretrained_model_name_or_path (str): HuggingFace model ID or local path.
            Defaults to the official 200M checkpoint.
        pooling (str): How to reduce the patch sequence to a single vector.
            - "mean"  : average over all patch positions  (default)
            - "last"  : take the last patch token
            - "none"  : return the full hidden-state tensor (batch, patches, hidden)
        device_map (str | dict | None): Passed straight to from_pretrained.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "google/timesfm-2.5-200m-transformers",
        pooling: str = "mean",
        device_map=None,
    ):
        super().__init__()

        if pooling not in ("mean", "last", "none"):
            raise ValueError(f"pooling must be 'mean', 'last', or 'none', got '{pooling}'")
        self.pooling = pooling

        self.model = TimesFm2_5Model.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device_map,
        )
        self.model.eval()

        self.hidden_size: int = self.model.config.hidden_size
        self.patch_length: int = getattr(self.model.config, "patch_length", 32)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _to_tensor(self, ts) -> torch.Tensor:
        """Accept numpy arrays, plain lists, or tensors; return a 1-D float32 tensor."""
        if isinstance(ts, torch.Tensor):
            return ts.float()
        return torch.tensor(ts, dtype=torch.float32)

    def _build_batch(
        self, time_series: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a list of variable-length 1-D series into a (B, T) batch tensor
        and a corresponding padding mask (1 = padded, 0 = valid).
        """
        tensors = [self._to_tensor(ts) for ts in time_series]
        max_len = max(t.shape[0] for t in tensors)
        pad_len = ((max_len + self.patch_length - 1) // self.patch_length) * self.patch_length

        padded = torch.zeros(len(tensors), pad_len)
        mask = torch.ones(len(tensors), pad_len, dtype=torch.long)  # 1 = padded

        for i, t in enumerate(tensors):
            length = t.shape[0]
            padded[i, :length] = t
            mask[i, :length] = 0  # valid positions

        return padded.to(self.device), mask.to(self.device)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        past_values: list,
        past_values_padding: torch.LongTensor | None = None,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            past_values: List of 1-D array-like time series (numpy / list / tensor).
                         They may differ in length; shorter ones are right-padded.
            past_values_padding: Optional explicit padding mask of shape
                                 (batch, max_seq_len).  If None, it is inferred
                                 automatically from the lengths of the inputs.
            output_hidden_states: If True, all intermediate layer hidden states
                                  are included in the returned dict.

        Returns:
            dict with keys:
                "embeddings"     – pooled or full last-layer hidden states
                                   shape depends on `pooling`:
                                     "mean"/"last" → (batch, hidden_size)
                                     "none"        → (batch, num_patches, hidden_size)
                "last_hidden_state" – raw (batch, num_patches, hidden_size) tensor
                "hidden_states"  – tuple of all layer outputs (if requested)
        """
        padded, auto_mask = self._build_batch(past_values)

        # Use caller-supplied mask only if provided; otherwise use the inferred one.
        mask = past_values_padding if past_values_padding is not None else auto_mask

        with torch.no_grad():
            outputs = self.model(
                past_values=padded,
                past_values_padding=mask,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        last_hidden: torch.Tensor = outputs.last_hidden_state  # (B, P, H)

        if self.pooling == "mean":
            embeddings = last_hidden.mean(dim=1)        # (B, H)
        elif self.pooling == "last":
            embeddings = last_hidden[:, -1, :]          # (B, H)
        else:  # "none"
            embeddings = last_hidden                    # (B, P, H)

        result = {
            "embeddings": embeddings,
            "last_hidden_state": last_hidden,
        }
        if output_hidden_states and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states

        return result