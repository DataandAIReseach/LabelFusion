#!/usr/bin/env python3
"""Test script to verify LLM results saving functionality with mock data."""

import pandas as pd
import sys
import os
from pathlib import Path

# Add the package to the path
sys.path.insert(0, '.')

from textclassify.core.types import ModelConfig, ClassificationResult, ClassificationType, ModelType
from textclassify.utils.results_manager import ResultsManager, ModelResultsManager

def test_llm_results_manager():
    """Test the results manager functionality for LLM classifiers."""
    
    print("🧠 Testing LLM Results Manager functionality...")
    
    # Create sample data
    test_data = {
        'text': [
            "This is a positive review",
            "This product is terrible", 
            "Great service and quality"
        ],
        'label': ['positive', 'negative', 'positive']
    }
    test_df = pd.DataFrame(test_data)
    
    # Create mock LLM predictions
    mock_predictions = ['positive', 'negative', 'positive']
    mock_metrics = {
        'accuracy': 0.95,
        'precision': 0.93,
        'recall': 0.97,
        'f1': 0.95
    }
    
    try:
        # Create results manager for LLM
        results_manager = ResultsManager(
            base_output_dir="outputs",
            experiment_name="test_openai_gpt_3_5_turbo"
        )
        
        model_results_manager = ModelResultsManager(
            results_manager,
            f"openai_classifier_{results_manager.experiment_id}"
        )
        
        print(f"📂 Experiment ID: {results_manager.experiment_id}")
        
        # Create a ClassificationResult object
        llm_result = ClassificationResult(
            predictions=mock_predictions,
            model_name="gpt-3.5-turbo",
            model_type=ModelType.LLM,
            classification_type=ClassificationType.SINGLE_LABEL,
            metadata={'metrics': mock_metrics}
        )
        
        # Save predictions
        print("💾 Saving LLM predictions...")
        saved_files = results_manager.save_predictions(
            llm_result, "test", test_df
        )
        print(f"📄 Prediction files: {saved_files}")
        
        # Save metrics
        print("📊 Saving LLM metrics...")
        metrics_file = results_manager.save_metrics(
            mock_metrics, "test", "openai_classifier"
        )
        print(f"📈 Metrics file: {metrics_file}")
        
        # Save model configuration
        print("⚙️ Saving LLM configuration...")
        llm_config = {
            'provider': 'openai',
            'model_name': 'gpt-3.5-turbo',
            'multi_label': False,
            'few_shot_mode': 'few_shot',
            'text_column': 'text',
            'label_columns': ['positive', 'negative'],
            'temperature': 0.1,
            'max_completion_tokens': 150
        }
        
        config_file = results_manager.save_model_config(
            llm_config, "openai_classifier"
        )
        print(f"⚙️ Config file: {config_file}")
        
        # Save experiment summary
        print("📋 Saving experiment summary...")
        experiment_summary = {
            'model_type': 'llm',
            'provider': 'openai',
            'model_name': 'gpt-3.5-turbo',
            'test_samples': len(test_df),
            'accuracy': mock_metrics['accuracy'],
            'completed': True
        }
        
        results_manager.save_experiment_summary(experiment_summary)
        
        # Get experiment info
        exp_info = results_manager.get_experiment_info()
        print(f"📂 Experiment directory: {exp_info['experiment_dir']}")
        
        # Check what files were actually created
        experiment_dir = Path(exp_info['experiment_dir'])
        if experiment_dir.exists():
            print(f"✅ Experiment directory exists: {experiment_dir}")
            
            # List all files in the experiment
            for subdir in ['predictions', 'metrics', 'models', 'logs']:
                subdir_path = experiment_dir / subdir
                if subdir_path.exists():
                    files = list(subdir_path.glob('*'))
                    print(f"📁 {subdir}/: {[f.name for f in files]}")
                else:
                    print(f"📁 {subdir}/: (empty)")
            
            # Check for experiment summary
            summary_file = experiment_dir / "experiment_summary.yaml"
            if summary_file.exists():
                print(f"📋 experiment_summary.yaml: ✅")
            else:
                print(f"📋 experiment_summary.yaml: ❌")
        else:
            print(f"❌ Experiment directory not found: {experiment_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 Testing LLM Results Manager Functionality")
    print("=" * 60)
    
    success = test_llm_results_manager()
    
    if success:
        print("\n✅ LLM results manager test completed successfully!")
        print("🎯 This demonstrates that LLM classifiers would save results")
        print("   to the experiments directory when predictions are made.")
    else:
        print("\n❌ LLM results manager test failed!")