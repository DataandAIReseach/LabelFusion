"""
Comprehensive test suite for the timeseries module.

Tests TSEmbedder, TSEmbeddingFuser, and CrossTSTransformer.

Usage:
    python test_timeseries.py
"""

import sys
import numpy as np
import torch
import pytest
from typing import Dict, List
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the timeseries module under test
from textclassify.timeseries.TSEmbedder import TSEmbedder
from textclassify.timeseries.TSEmbeddingFuser import TSEmbeddingFuser
from textclassify.timeseries.CrossTSTransformer import CrossTSTransformer


class TestTSEmbedder:
    """Test suite for TSEmbedder."""
    
    @pytest.fixture
    def embedder(self):
        """Create a TSEmbedder instance for testing."""
        return TSEmbedder(
            pretrained_model_name_or_path="google/timesfm-2.5-200m-transformers",
            pooling="mean",
            device_map="auto"
        )
    
    def test_initialization(self, embedder):
        """Test TSEmbedder initialization."""
        assert embedder.pooling == "mean"
        assert embedder.hidden_size == 1280  # TimesFM 2.5 hidden size
        assert embedder.model is not None
    
    def test_invalid_pooling(self):
        """Test that invalid pooling modes raise ValueError."""
        with pytest.raises(ValueError, match="pooling must be"):
            TSEmbedder(pooling="invalid")
    
    def test_single_series_embedding(self, embedder):
        """Test embedding a single time series."""
        series = [np.sin(np.linspace(0, 20, 100))]
        result = embedder(series)
        
        assert "embeddings" in result
        assert "last_hidden_state" in result
        assert result["embeddings"].shape == (1, 1280)
        assert len(result["last_hidden_state"].shape) == 3  # (B, P, H)
    
    def test_batch_embedding(self, embedder):
        """Test embedding multiple time series of different lengths."""
        series = [
            np.sin(np.linspace(0, 20, 100)),
            np.sin(np.linspace(0, 20, 200)),
            np.sin(np.linspace(0, 20, 400)),
        ]
        result = embedder(series)
        
        assert result["embeddings"].shape == (3, 1280)
        assert result["last_hidden_state"].shape[0] == 3
    
    def test_pooling_modes(self):
        """Test different pooling modes."""
        series = [np.sin(np.linspace(0, 20, 100))]
        
        # Mean pooling
        embedder_mean = TSEmbedder(pooling="mean", device_map="auto")
        result_mean = embedder_mean(series)
        assert result_mean["embeddings"].shape == (1, 1280)
        
        # Last pooling
        embedder_last = TSEmbedder(pooling="last", device_map="auto")
        result_last = embedder_last(series)
        assert result_last["embeddings"].shape == (1, 1280)
        
        # None pooling (returns full sequence)
        embedder_none = TSEmbedder(pooling="none", device_map="auto")
        result_none = embedder_none(series)
        assert len(result_none["embeddings"].shape) == 3  # (B, P, H)
    
    def test_to_tensor_conversion(self, embedder):
        """Test _to_tensor handles different input types."""
        # Numpy array
        np_array = np.array([1.0, 2.0, 3.0])
        tensor = embedder._to_tensor(np_array)
        assert isinstance(tensor, torch.Tensor)
        
        # List
        list_input = [1.0, 2.0, 3.0]
        tensor = embedder._to_tensor(list_input)
        assert isinstance(tensor, torch.Tensor)
        
        # Already a tensor
        existing_tensor = torch.tensor([1.0, 2.0, 3.0])
        tensor = embedder._to_tensor(existing_tensor)
        assert isinstance(tensor, torch.Tensor)
    
    def test_build_batch_padding(self, embedder):
        """Test _build_batch creates proper padding."""
        series = [
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2]),
        ]
        
        padded, mask = embedder._build_batch(series)
        
        assert padded.shape == (3, 5)  # Max length is 5
        assert mask.shape == (3, 5)
        
        # Check padding mask (1 = padded, 0 = valid)
        assert mask[0, 3] == 1  # Series 1 padded from position 3
        assert mask[0, 0] == 0  # Series 1 valid at position 0
        assert mask[1, 4] == 0  # Series 2 valid to position 4
        assert mask[2, 2] == 1  # Series 3 padded from position 2


class TestTSEmbeddingFuser:
    """Test suite for TSEmbeddingFuser."""
    
    @pytest.fixture
    def series_names(self):
        """Sample commodity names."""
        return ["gold", "silver", "crude"]
    
    @pytest.fixture
    def fuser(self, series_names):
        """Create a TSEmbeddingFuser instance."""
        return TSEmbeddingFuser(
            series_names=series_names,
            pooling="mean",
            device_map="auto"
        )
    
    def test_initialization(self, fuser, series_names):
        """Test TSEmbeddingFuser initialization."""
        assert fuser.series_names == series_names
        assert fuser.hidden_size == 1280
        assert fuser.stacked_size == 3 * 1280  # 3 series × 1280
        assert fuser.embedder is not None
    
    def test_empty_series_names_raises_error(self):
        """Test that empty series_names raises ValueError."""
        with pytest.raises(ValueError, match="series_names must not be empty"):
            TSEmbeddingFuser(series_names=[])
    
    def test_pooling_none_raises_error(self):
        """Test that pooling='none' raises ValueError."""
        with pytest.raises(ValueError, match="pooling='none'"):
            TSEmbeddingFuser(series_names=["gold"], pooling="none")
    
    def test_forward_single_batch(self, fuser):
        """Test forward pass with a single batch."""
        series_dict = {
            "gold": [np.sin(np.linspace(0, 20, 100))],
            "silver": [np.cos(np.linspace(0, 20, 100))],
            "crude": [np.sin(np.linspace(0, 10, 50))],
        }
        
        result = fuser(series_dict)
        
        assert "stacked" in result
        assert "per_series" in result
        assert "stacked_size" in result
        
        assert result["stacked"].shape == (1, 3840)  # (1, 3 × 1280)
        assert len(result["per_series"]) == 3
        assert result["per_series"]["gold"].shape == (1, 1280)
    
    def test_forward_multiple_batches(self, fuser):
        """Test forward pass with multiple batches."""
        batch_size = 5
        series_dict = {
            "gold": [np.sin(np.linspace(0, 20, 100)) for _ in range(batch_size)],
            "silver": [np.cos(np.linspace(0, 20, 100)) for _ in range(batch_size)],
            "crude": [np.sin(np.linspace(0, 10, 50)) for _ in range(batch_size)],
        }
        
        result = fuser(series_dict)
        
        assert result["stacked"].shape == (5, 3840)
        assert result["per_series"]["gold"].shape == (5, 1280)
    
    def test_missing_series_raises_error(self, fuser):
        """Test that missing series in input raises KeyError."""
        series_dict = {
            "gold": [np.sin(np.linspace(0, 20, 100))],
            "silver": [np.cos(np.linspace(0, 20, 100))],
            # "crude" is missing
        }
        
        with pytest.raises(KeyError, match="Missing series"):
            fuser(series_dict)
    
    def test_inconsistent_batch_size_raises_error(self, fuser):
        """Test that inconsistent batch sizes raise ValueError."""
        series_dict = {
            "gold": [np.sin(np.linspace(0, 20, 100))],
            "silver": [np.cos(np.linspace(0, 20, 100)), np.cos(np.linspace(0, 20, 100))],  # 2 samples
            "crude": [np.sin(np.linspace(0, 10, 50))],
        }
        
        with pytest.raises(ValueError, match="expected"):
            fuser(series_dict)
    
    def test_custom_embedder(self, series_names):
        """Test using a custom pre-built embedder."""
        custom_embedder = TSEmbedder(pooling="last", device_map="auto")
        fuser = TSEmbeddingFuser(series_names=series_names, embedder=custom_embedder)
        
        assert fuser.embedder is custom_embedder
        assert fuser.embedder.pooling == "last"


class TestCrossTSTransformer:
    """Test suite for CrossTSTransformer."""
    
    @pytest.fixture
    def series_names(self):
        """Sample commodity names."""
        return ["gold", "silver", "crude", "wheat", "corn"]
    
    @pytest.fixture
    def fuser(self, series_names):
        """Create TSEmbeddingFuser for CrossTSTransformer."""
        return TSEmbeddingFuser(
            series_names=series_names,
            pooling="mean",
            device_map="auto"
        )
    
    @pytest.fixture
    def transformer(self, fuser):
        """Create CrossTSTransformer instance."""
        return CrossTSTransformer(
            stacked_embedder=fuser,
            output_dim=256,
            num_heads=8,
            num_layers=2,
            dropout=0.1,
            feedforward_dim=2048
        )
    
    def test_initialization(self, transformer, series_names):
        """Test CrossTSTransformer initialization."""
        assert transformer.hidden_size == 1280
        assert transformer.n_series == 5
        assert transformer.output_dim == 256
        assert transformer.pos_embedding is not None
        assert transformer.transformer is not None
        assert transformer.projection is not None
    
    def test_forward_single_batch(self, transformer, series_names):
        """Test forward pass with single batch."""
        series_dict = {
            name: [np.sin(np.linspace(0, 20, 100))]
            for name in series_names
        }
        
        result = transformer(series_dict)
        
        assert "output" in result
        assert "pooled" in result
        assert "attended" in result
        assert "per_series" in result
        
        assert result["output"].shape == (1, 256)
        assert result["pooled"].shape == (1, 1280)
        assert result["attended"].shape == (1, 5, 1280)
        assert len(result["per_series"]) == 5
    
    def test_forward_multiple_batches(self, transformer, series_names):
        """Test forward pass with multiple batches."""
        batch_size = 8
        series_dict = {
            name: [np.sin(np.linspace(0, 20, 100)) for _ in range(batch_size)]
            for name in series_names
        }
        
        result = transformer(series_dict)
        
        assert result["output"].shape == (8, 256)
        assert result["pooled"].shape == (8, 1280)
        assert result["attended"].shape == (8, 5, 1280)
    
    def test_positional_embedding(self, transformer):
        """Test positional embeddings are added correctly."""
        # The pos_embedding should have one entry per series
        assert transformer.pos_embedding.num_embeddings == transformer.n_series
        assert transformer.pos_embedding.embedding_dim == transformer.hidden_size
    
    def test_cross_series_attention(self, transformer, series_names):
        """Test that cross-series attention captures interactions."""
        # Create two batches with different patterns
        series_dict_1 = {
            "gold": [np.ones(100)],
            "silver": [np.ones(100)],
            "crude": [np.zeros(100)],
            "wheat": [np.zeros(100)],
            "corn": [np.zeros(100)],
        }
        
        series_dict_2 = {
            "gold": [np.zeros(100)],
            "silver": [np.zeros(100)],
            "crude": [np.ones(100)],
            "wheat": [np.ones(100)],
            "corn": [np.ones(100)],
        }
        
        result_1 = transformer(series_dict_1)
        result_2 = transformer(series_dict_2)
        
        # Outputs should be different due to different input patterns
        assert not torch.allclose(result_1["output"], result_2["output"])
    
    def test_different_output_dims(self, fuser):
        """Test creating transformers with different output dimensions."""
        for output_dim in [128, 256, 512]:
            transformer = CrossTSTransformer(
                stacked_embedder=fuser,
                output_dim=output_dim
            )
            
            series_dict = {
                name: [np.sin(np.linspace(0, 20, 100))]
                for name in fuser.series_names
            }
            
            result = transformer(series_dict)
            assert result["output"].shape == (1, output_dim)
    
    def test_gradient_flow(self, transformer, series_names):
        """Test that gradients flow through the transformer."""
        transformer.train()  # Set to training mode
        
        series_dict = {
            name: [np.sin(np.linspace(0, 20, 100))]
            for name in series_names
        }
        
        result = transformer(series_dict)
        output = result["output"]
        
        # Create dummy target and loss
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Check gradients can be computed
        loss.backward()
        
        # Check that parameters have gradients
        has_gradients = any(
            p.grad is not None
            for p in transformer.parameters()
            if p.requires_grad
        )
        assert has_gradients


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_10_stocks(self):
        """Test the full pipeline with 10 stock symbols (FNSPID dataset)."""
        # FNSPID top 10 stock symbols from 2022
        stocks = ["AAPL", "AMD", "BRK", "DIS", "GOOG", "MSFT", "NVDA", "TSLA", "WMT", "XOM"]
        
        # Create pipeline
        fuser = TSEmbeddingFuser(
            series_names=stocks,
            pooling="mean",
            device_map="auto"
        )
        
        transformer = CrossTSTransformer(
            stacked_embedder=fuser,
            output_dim=256,
            num_heads=8,
            num_layers=2
        )
        
        # Create batch of price data
        batch_size = 16
        series_dict = {
            name: [np.random.randn(100) for _ in range(batch_size)]
            for name in stocks
        }
        
        # Forward pass
        result = transformer(series_dict)
        
        # Verify output shapes
        assert result["output"].shape == (batch_size, 256)
        assert result["attended"].shape == (batch_size, len(stocks), 1280)
    
    def test_pipeline_with_varying_series_lengths(self):
        """Test pipeline handles time series of different lengths."""
        commodities = ["gold", "silver", "crude"]
        
        fuser = TSEmbeddingFuser(series_names=commodities, pooling="mean")
        transformer = CrossTSTransformer(stacked_embedder=fuser, output_dim=128)
        
        # Different lengths per series
        series_dict = {
            "gold": [np.random.randn(50), np.random.randn(150)],
            "silver": [np.random.randn(100), np.random.randn(200)],
            "crude": [np.random.randn(75), np.random.randn(125)],
        }
        
        result = transformer(series_dict)
        assert result["output"].shape == (2, 128)
    
    def test_end_to_end_fnspid_simulation(self):
        """Simulate the FNSPID stock classification use case with 10 stock symbols."""
        # FNSPID top 10 stock symbols: Apple, AMD, Berkshire, Disney, Google, Microsoft, NVIDIA, Tesla, Walmart, Exxon
        stocks = ["AAPL", "AMD", "BRK", "DIS", "GOOG", "MSFT", "NVDA", "TSLA", "WMT", "XOM"]
        batch_size = 4  # 4 articles
        
        # Create embedder and transformer
        fuser = TSEmbeddingFuser(series_names=stocks, pooling="mean")
        ts_branch = CrossTSTransformer(
            stacked_embedder=fuser,
            output_dim=256
        )
        
        # Simulate price data for each article's publication date
        series_dict = {
            name: [np.sin(np.linspace(0, 20, 100)) for _ in range(batch_size)]
            for name in stocks
        }
        
        # Get TS representation
        ts_output = ts_branch(series_dict)["output"]  # (4, 256)
        
        # Simulate other branches
        roberta_output = torch.randn(batch_size, 768)  # RoBERTa embeddings
        llm_output = torch.randint(0, 2, (batch_size, 10)).float()  # LLM 0/1 predictions (10 stocks)
        
        # Concatenate all branches (simulating FusionMLP input)
        fusion_input = torch.cat([roberta_output, llm_output, ts_output], dim=1)
        
        assert fusion_input.shape == (batch_size, 768 + 10 + 256)  # (4, 1034)


def run_tests():
    """Run all tests."""
    print("=" * 70)
    print("TIMESERIES MODULE TEST SUITE")
    print("=" * 70)
    
    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()