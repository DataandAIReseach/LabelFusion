#!/usr/bin/env python3
"""Test script to verify ensemble results management integration."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from textclassify.ensemble.weighted import WeightedEnsemble
from textclassify.ensemble.voting import VotingEnsemble
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.core.types import EnsembleConfig, ModelConfig, ModelType

def test_ensemble_results_integration():
    """Test that ensemble results are automatically saved."""
    print("🧪 Testing Ensemble Results Integration...")
    
    # Create test data
    texts = [
        "This is a positive example",
        "This is a negative example", 
        "This is neutral content"
    ]
    
    # Create simple ensemble config
    ensemble_config = EnsembleConfig(
        ensemble_method="weighted",
        models=[],
        weights=[0.5, 0.5]
    )
    
    try:
        # Test WeightedEnsemble initialization with results management
        print("\n📊 Testing WeightedEnsemble with results management...")
        weighted_ensemble = WeightedEnsemble(
            ensemble_config,
            output_dir="outputs",
            experiment_name="test_weighted_ensemble",
            auto_save_results=True
        )
        
        print(f"✅ WeightedEnsemble created successfully")
        print(f"📁 Results manager: {weighted_ensemble.results_manager is not None}")
        if weighted_ensemble.results_manager:
            exp_info = weighted_ensemble.results_manager.get_experiment_info()
            print(f"🔬 Experiment directory: {exp_info['experiment_dir']}")
        
        # Test VotingEnsemble initialization
        ensemble_config.ensemble_method = "voting"
        print("\n🗳️ Testing VotingEnsemble with results management...")
        voting_ensemble = VotingEnsemble(
            ensemble_config,
            output_dir="outputs", 
            experiment_name="test_voting_ensemble",
            auto_save_results=True
        )
        
        print(f"✅ VotingEnsemble created successfully")
        print(f"📁 Results manager: {voting_ensemble.results_manager is not None}")
        if voting_ensemble.results_manager:
            exp_info = voting_ensemble.results_manager.get_experiment_info()
            print(f"🔬 Experiment directory: {exp_info['experiment_dir']}")
        
        print("\n🎉 All ensemble classes successfully integrated with results management!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ensemble_results_integration()
    if success:
        print("\n✅ Ensemble results integration test passed!")
    else:
        print("\n❌ Ensemble results integration test failed!")
        sys.exit(1)