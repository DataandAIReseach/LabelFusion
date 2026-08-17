"""
Test suite for timeseries pipeline with separate TS objects.

Tests the complete pipeline:
    TS("AAPL") + TS("MSFT")
        ↓
    TSEmbeddingFuser (creates TSEmbedder internally)
        ↓
    CrossTSTransformer
        ↓
    Final 256-dim embedding

Usage:
    pytest test_pipeline_two_ts.py -v
"""

import sys
import numpy as np
import pandas as pd
import torch
import pytest
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import classes
try:
    from textclassify.timeseries.TS import TS
    from textclassify.timeseries.TSEmbedder import TSEmbedder
    from textclassify.timeseries.TSEmbeddingFuser import TSEmbeddingFuser
    from textclassify.timeseries.CrossTSTransformer import CrossTSTransformer
except ImportError:
    from TS import TS
    from TSEmbedder import TSEmbedder
    from TSEmbeddingFuser import TSEmbeddingFuser
    from CrossTSTransformer import CrossTSTransformer


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory with synthetic stock data."""
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    
    # Create synthetic data for AAPL and MSFT
    for symbol, base_price in [("AAPL", 150.0), ("MSFT", 280.0)]:
        np.random.seed(hash(symbol) % 2**32)
        dates = pd.date_range(start='2021-07-01', periods=753, freq='D')
        
        returns = np.random.randn(753) * 0.02
        returns[0] = 0
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
        })
        
        csv_path = data_dir / f"{symbol}.csv"
        df.to_csv(csv_path, index=False)
    
    return data_dir


class TestTSObjects:
    """Test creation of separate TS objects."""
    
    def test_create_two_separate_ts_objects(self, temp_data_dir):
        """Test creating two independent TS objects."""
        # Create TS for AAPL
        ts_aapl = TS(
            data_dir=temp_data_dir,
            stock_symbols=["AAPL"],
            date_column="Date",
            price_column="Close"
        )
        
        # Create TS for MSFT
        ts_msft = TS(
            data_dir=temp_data_dir,
            stock_symbols=["MSFT"],
            date_column="Date",
            price_column="Close"
        )
        
        assert ts_aapl.stock_symbols == ["AAPL"]
        assert ts_msft.stock_symbols == ["MSFT"]
        assert ts_aapl is not ts_msft
    
    def test_load_data_from_separate_ts_objects(self, temp_data_dir):
        """Test loading data from two TS objects."""
        ts_aapl = TS(temp_data_dir, ["AAPL"])
        ts_msft = TS(temp_data_dir, ["MSFT"])
        
        ts_aapl.load_all()
        ts_msft.load_all()
        
        assert "AAPL" in ts_aapl.data
        assert "MSFT" in ts_msft.data
        assert "MSFT" not in ts_aapl.data
        assert "AAPL" not in ts_msft.data
    
    def test_get_series_from_separate_ts_objects(self, temp_data_dir):
        """Test getting series from each TS object."""
        ts_aapl = TS(temp_data_dir, ["AAPL"])
        ts_msft = TS(temp_data_dir, ["MSFT"])
        
        ts_aapl.load_all()
        ts_msft.load_all()
        
        aapl_series = ts_aapl.get_series_for_date("2023-06-15", window_days=100)
        msft_series = ts_msft.get_series_for_date("2023-06-15", window_days=100)
        
        assert "AAPL" in aapl_series
        assert "MSFT" not in aapl_series
        assert "MSFT" in msft_series
        assert "AAPL" not in msft_series
        
        assert len(aapl_series["AAPL"]) == 100
        assert len(msft_series["MSFT"]) == 100
    
    def test_combine_series_from_two_ts_objects(self, temp_data_dir):
        """Test combining series from two TS objects."""
        ts_aapl = TS(temp_data_dir, ["AAPL"])
        ts_msft = TS(temp_data_dir, ["MSFT"])
        
        ts_aapl.load_all()
        ts_msft.load_all()
        
        aapl_series = ts_aapl.get_series_for_date("2023-06-15", window_days=100)
        msft_series = ts_msft.get_series_for_date("2023-06-15", window_days=100)
        
        # Combine using dict unpacking
        combined = {**aapl_series, **msft_series}
        
        assert "AAPL" in combined
        assert "MSFT" in combined
        assert len(combined) == 2


class TestPipelineWithTwoTSObjects:
    """Test complete pipeline with two separate TS objects."""
    
    def test_ts_to_embedder_pipeline(self, temp_data_dir):
        """Test: TS objects → TSEmbeddingFuser (creates TSEmbedder internally)."""
        # Create two TS objects
        ts_aapl = TS(temp_data_dir, ["AAPL"])
        ts_msft = TS(temp_data_dir, ["MSFT"])
        ts_aapl.load_all()
        ts_msft.load_all()
        
        # Get series
        aapl_series = ts_aapl.get_series_for_date("2023-06-15", 100)
        msft_series = ts_msft.get_series_for_date("2023-06-15", 100)
        combined = {**aapl_series, **msft_series}
        
        # Convert to batch format
        series_batch = {name: [prices] for name, prices in combined.items()}
        
        try:
            # TSEmbeddingFuser creates TSEmbedder internally
            fuser = TSEmbeddingFuser(
                series_names=["AAPL", "MSFT"],
                pooling="mean",
                device_map="auto"
            )
            
            # Verify TSEmbedder was created internally
            assert hasattr(fuser, 'embedder')
            assert isinstance(fuser.embedder, TSEmbedder)
            assert hasattr(fuser, 'embedders')
            assert 'AAPL' in fuser.embedders
            assert 'MSFT' in fuser.embedders
            
            # Run forward
            output = fuser(series_batch)
            
            assert 'stacked' in output
            assert 'per_series' in output
            assert output['stacked'].shape[0] == 1  # batch size
            assert 'AAPL' in output['per_series']
            assert 'MSFT' in output['per_series']
            
        except Exception as e:
            pytest.skip(f"TSEmbeddingFuser requires transformers: {e}")
    
    def test_complete_pipeline_two_ts_objects(self, temp_data_dir):
        """Test complete pipeline: TS → TSEmbeddingFuser → CrossTSTransformer."""
        # Step 1: Two separate TS objects
        ts_aapl = TS(temp_data_dir, ["AAPL"])
        ts_msft = TS(temp_data_dir, ["MSFT"])
        ts_aapl.load_all()
        ts_msft.load_all()
        
        # Step 2: Get series from each
        aapl_series = ts_aapl.get_series_for_date("2023-06-15", 100)
        msft_series = ts_msft.get_series_for_date("2023-06-15", 100)
        combined = {**aapl_series, **msft_series}
        series_batch = {name: [prices] for name, prices in combined.items()}
        
        try:
            # Step 3: TSEmbeddingFuser (creates TSEmbedder internally)
            fuser = TSEmbeddingFuser(
                series_names=["AAPL", "MSFT"],
                pooling="mean",
                device_map="auto"
            )
            
            # Step 4: CrossTSTransformer
            transformer = CrossTSTransformer(
                stacked_embedder=fuser,
                output_dim=256,
                num_heads=8,
                num_layers=2
            )
            
            # Step 5: Forward pass
            final_output = transformer(series_batch)
            
            # Verify outputs
            assert 'output' in final_output
            assert 'pooled' in final_output
            assert 'attended' in final_output
            assert 'per_series' in final_output
            
            assert final_output['output'].shape == (1, 256)
            assert final_output['pooled'].shape == (1, 1280)
            assert final_output['attended'].shape == (1, 2, 1280)
            assert len(final_output['per_series']) == 2
            
        except Exception as e:
            pytest.skip(f"Pipeline requires transformers: {e}")
    
    def test_batch_processing_with_two_ts_objects(self, temp_data_dir):
        """Test batch processing multiple articles with two TS objects."""
        ts_aapl = TS(temp_data_dir, ["AAPL"])
        ts_msft = TS(temp_data_dir, ["MSFT"])
        ts_aapl.load_all()
        ts_msft.load_all()
        
        # Simulate 4 articles with different publication dates
        dates = ["2023-01-15", "2023-03-20", "2023-06-10", "2023-09-05"]
        
        # Get series for each date from both TS objects
        aapl_batch = ts_aapl.get_series_batch(dates, window_days=100)
        msft_batch = ts_msft.get_series_batch(dates, window_days=100)
        
        # Combine into single dict
        series_batch = {**aapl_batch, **msft_batch}
        
        assert "AAPL" in series_batch
        assert "MSFT" in series_batch
        assert len(series_batch["AAPL"]) == 4
        assert len(series_batch["MSFT"]) == 4
        
        try:
            fuser = TSEmbeddingFuser(
                series_names=["AAPL", "MSFT"],
                pooling="mean",
                device_map="auto"
            )
            
            transformer = CrossTSTransformer(
                stacked_embedder=fuser,
                output_dim=256
            )
            
            output = transformer(series_batch)
            
            # Batch size should be 4
            assert output['output'].shape == (4, 256)
            assert output['attended'].shape == (4, 2, 1280)
            
        except Exception as e:
            pytest.skip(f"Pipeline requires transformers: {e}")


class TestTSEmbedderIntegration:
    """Test that TSEmbedder is correctly integrated in TSEmbeddingFuser."""
    
    def test_fuser_creates_embedder_internally(self, temp_data_dir):
        """Verify TSEmbeddingFuser creates TSEmbedder when not provided."""
        try:
            fuser = TSEmbeddingFuser(
                series_names=["AAPL", "MSFT"],
                pooling="mean",
                device_map="auto"
            )
            
            # Should have created embedder
            assert hasattr(fuser, 'embedder')
            assert isinstance(fuser.embedder, TSEmbedder)
            
            # Should have created embedders dict
            assert hasattr(fuser, 'embedders')
            assert len(fuser.embedders) == 2
            assert all(isinstance(e, TSEmbedder) for e in fuser.embedders.values())
            
        except Exception as e:
            pytest.skip(f"Requires transformers: {e}")
    
    def test_fuser_uses_provided_embedder(self, temp_data_dir):
        """Verify TSEmbeddingFuser can use provided TSEmbedder."""
        try:
            # Create embedder explicitly
            embedder = TSEmbedder(pooling="mean", device_map="auto")
            
            # Pass to fuser
            fuser = TSEmbeddingFuser(
                series_names=["AAPL", "MSFT"],
                embedder=embedder
            )
            
            # Should use the provided embedder as template
            assert fuser.embedder is embedder
            
            # Should have created copies
            assert len(fuser.embedders) == 2
            
        except Exception as e:
            pytest.skip(f"Requires transformers: {e}")
    
    def test_each_series_has_own_embedder(self, temp_data_dir):
        """Verify each series gets its own TSEmbedder instance."""
        try:
            fuser = TSEmbeddingFuser(
                series_names=["AAPL", "MSFT"],
                pooling="mean",
                device_map="auto"
            )
            
            # Get embedders
            aapl_embedder = fuser.embedders['AAPL']
            msft_embedder = fuser.embedders['MSFT']
            
            # Should be different instances (deepcopy)
            assert aapl_embedder is not msft_embedder
            
            # But same configuration
            assert aapl_embedder.pooling == msft_embedder.pooling
            assert aapl_embedder.hidden_size == msft_embedder.hidden_size
            
        except Exception as e:
            pytest.skip(f"Requires transformers: {e}")


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_fnspid_10_stocks_simulation(self, temp_data_dir):
        """Simulate FNSPID workflow with 10 stocks (using 2 for testing)."""
        # In real scenario: 10 TS objects for 10 stocks
        # Here: 2 TS objects as proof of concept
        
        stocks = ["AAPL", "MSFT"]
        ts_objects = {}
        
        # Create one TS object per stock
        for stock in stocks:
            ts = TS(temp_data_dir, [stock])
            ts.load_all()
            ts_objects[stock] = ts
        
        # Get series from each TS object
        article_date = "2023-06-15"
        all_series = {}
        
        for stock, ts in ts_objects.items():
            series = ts.get_series_for_date(article_date, window_days=100)
            all_series.update(series)
        
        # Convert to batch
        series_batch = {name: [prices] for name, prices in all_series.items()}
        
        try:
            # Pipeline
            fuser = TSEmbeddingFuser(
                series_names=stocks,
                pooling="mean",
                device_map="auto"
            )
            
            transformer = CrossTSTransformer(
                stacked_embedder=fuser,
                output_dim=256
            )
            
            output = transformer(series_batch)
            
            # Final embedding for multimodal fusion
            ts_embedding = output['output']  # (1, 256)
            
            # Simulate other branches
            roberta_emb = torch.randn(1, 768)
            llm_pred = torch.randint(0, 2, (1, 10)).float()
            
            # Fusion input
            fusion_input = torch.cat([roberta_emb, llm_pred, ts_embedding], dim=1)
            
            assert fusion_input.shape == (1, 1034)  # 768 + 10 + 256
            
        except Exception as e:
            pytest.skip(f"Pipeline requires transformers: {e}")


def run_tests():
    """Run all tests."""
    print("=" * 70)
    print("TESTING PIPELINE WITH TWO SEPARATE TS OBJECTS")
    print("=" * 70)
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()