"""Quick mock to test ML model caching and persistence.

This demonstrates how ML models (RoBERTa) can be saved/loaded and how
embeddings and predictions are cached for reuse in fusion ensembles.

Run: python examples/ml_cache_mock.py
"""

import sys
from pathlib import Path
import pandas as pd
import json
import random
import tempfile
import shutil

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Simple mock for demonstrating ML caching patterns
class MockMLModel:
    """Mock ML model that simulates RoBERTa behavior with caching."""
    
    def __init__(self, model_name='roberta-base', cache_dir=None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path('cache/mock_ml')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.is_trained = False
        self.model_hash = None
        
    def train(self, train_df):
        """Simulate training and save model checkpoint."""
        print(f"Training mock ML model on {len(train_df)} samples...")
        self.is_trained = True
        # Generate deterministic hash based on training data
        text_concat = ''.join(train_df['text'].astype(str))
        self.model_hash = str(hash(text_concat) % 100000)
        
        # Save mock model checkpoint
        checkpoint = {
            'model_name': self.model_name,
            'model_hash': self.model_hash,
            'num_samples': len(train_df),
            'is_trained': True
        }
        checkpoint_path = self.cache_dir / f'model_{self.model_hash}.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        print(f"Saved model checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        self.model_hash = checkpoint['model_hash']
        self.is_trained = checkpoint['is_trained']
        print(f"Loaded model checkpoint: {checkpoint_path}")
        
    def predict_with_embeddings(self, test_df):
        """Generate predictions and embeddings (768-dim for RoBERTa)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        print(f"Generating predictions and embeddings for {len(test_df)} samples...")
        predictions = [[random.choice([0, 1])] for _ in range(len(test_df))]
        # Mock 768-dimensional embeddings (RoBERTa CLS token)
        embeddings = [[random.random() for _ in range(768)] for _ in range(len(test_df))]
        
        return {
            'predictions': predictions,
            'embeddings': embeddings,
            'model_hash': self.model_hash
        }
    
    def cache_predictions(self, test_df, result, dataset_type='test'):
        """Cache predictions and embeddings for reuse."""
        cache_path = self.cache_dir / f'{dataset_type}_predictions_{self.model_hash}.json'
        
        # Create cache with text hashes for lookup
        cache_data = {
            'model_hash': self.model_hash,
            'dataset_type': dataset_type,
            'num_samples': len(test_df),
            'predictions': {}
        }
        
        for idx, text in enumerate(test_df['text']):
            text_hash = str(hash(text) % 100000)
            cache_data['predictions'][text_hash] = {
                'prediction': result['predictions'][idx],
                'embedding': result['embeddings'][idx]
            }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        print(f"Cached {len(cache_data['predictions'])} predictions: {cache_path}")
        return cache_path
    
    def load_cached_predictions(self, test_df, dataset_type='test'):
        """Load cached predictions if available."""
        cache_path = self.cache_dir / f'{dataset_type}_predictions_{self.model_hash}.json'
        
        if not cache_path.exists():
            print(f"No cache found: {cache_path}")
            return None
        
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        print(f"Loaded cache: {cache_path}")
        
        # Reconstruct predictions and embeddings
        predictions = []
        embeddings = []
        cache_hits = 0
        
        for text in test_df['text']:
            text_hash = str(hash(text) % 100000)
            if text_hash in cache_data['predictions']:
                entry = cache_data['predictions'][text_hash]
                predictions.append(entry['prediction'])
                embeddings.append(entry['embedding'])
                cache_hits += 1
            else:
                predictions.append(None)
                embeddings.append(None)
        
        print(f"Cache hits: {cache_hits}/{len(test_df)}")
        return {
            'predictions': predictions,
            'embeddings': embeddings,
            'cache_hits': cache_hits
        }


def demo_ml_caching():
    """Demonstrate ML model caching workflow."""
    print("="*60)
    print("ML Model Caching Demo")
    print("="*60)
    
    # Create temporary cache directory
    cache_dir = project_root / 'cache' / 'mock_ml_demo'
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create toy datasets
    train_texts = [f"Training sample {i}" for i in range(10)]
    test_texts = [f"Test sample {i}" for i in range(5)]
    
    train_df = pd.DataFrame({'text': train_texts, 'label': [0, 1] * 5})
    test_df = pd.DataFrame({'text': test_texts, 'label': [0, 1, 0, 1, 0]})
    
    print(f"\nDatasets:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Step 1: Train model and save checkpoint
    print("\n" + "-"*60)
    print("Step 1: Train model")
    print("-"*60)
    model = MockMLModel(cache_dir=cache_dir)
    checkpoint_path = model.train(train_df)
    
    # Step 2: Generate predictions and cache them
    print("\n" + "-"*60)
    print("Step 2: Generate predictions and embeddings")
    print("-"*60)
    result = model.predict_with_embeddings(test_df)
    print(f"Predictions: {result['predictions']}")
    print(f"Embeddings shape: {len(result['embeddings'])} x {len(result['embeddings'][0])}")
    
    # Step 3: Cache predictions
    print("\n" + "-"*60)
    print("Step 3: Cache predictions and embeddings")
    print("-"*60)
    cache_path = model.cache_predictions(test_df, result, dataset_type='test')
    
    # Step 4: Simulate new session - load checkpoint
    print("\n" + "-"*60)
    print("Step 4: Simulate new session - load from checkpoint")
    print("-"*60)
    new_model = MockMLModel(cache_dir=cache_dir)
    new_model.load_checkpoint(checkpoint_path)
    
    # Step 5: Load cached predictions
    print("\n" + "-"*60)
    print("Step 5: Load cached predictions")
    print("-"*60)
    cached_result = new_model.load_cached_predictions(test_df, dataset_type='test')
    
    if cached_result and cached_result['cache_hits'] == len(test_df):
        print("\nCache verification:")
        print(f"  Original predictions: {result['predictions']}")
        print(f"  Cached predictions:   {cached_result['predictions']}")
        print(f"  Match: {result['predictions'] == cached_result['predictions']}")
    
    # Step 6: Show cache files
    print("\n" + "-"*60)
    print("Step 6: Cache directory contents")
    print("-"*60)
    cache_files = list(cache_dir.glob('*.json'))
    for f in cache_files:
        print(f"  {f.name} ({f.stat().st_size} bytes)")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print("\nKey concepts demonstrated:")
    print("  1. Model checkpointing (save/load trained models)")
    print("  2. Prediction caching (avoid re-inference)")
    print("  3. Embedding storage (768-dim vectors for fusion)")
    print("  4. Text-based hash lookup (fast cache retrieval)")
    print("  5. Cache hit tracking (verify cache usage)")


if __name__ == '__main__':
    demo_ml_caching()
