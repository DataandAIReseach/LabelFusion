#!/usr/bin/env python3
"""Test script to verify LLM results saving functionality."""

import pandas as pd
import sys
import os

# Add the package to the path
sys.path.insert(0, '.')

from textclassify.core.types import ModelConfig
from textclassify.llm.openai_classifier import OpenAIClassifier

def test_llm_results_saving():
    """Test that LLM classifier saves results to experiments directory."""
    
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
    
    # Simple train data for few-shot
    train_data = {
        'text': [
            "I love this product",
            "Worst purchase ever"
        ],
        'label': ['positive', 'negative']
    }
    train_df = pd.DataFrame(train_data)
    
    try:
        # Create configuration
        config = ModelConfig(
            model_type="llm",
            parameters={
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "max_completion_tokens": 50
            }
        )
        
        # Create LLM classifier with results saving enabled
        llm_classifier = OpenAIClassifier(
            config=config,
            text_column='text',
            label_columns=['positive', 'negative'],
            multi_label=False,
            auto_save_results=True,
            experiment_name="test_llm_saving"
        )
        
        print("🧠 Testing LLM classifier results saving...")
        print(f"Experiment directory should be created at: outputs/experiments/")
        
        # Make prediction (this should trigger results saving)
        result = llm_classifier.predict(
            train_df=train_df,
            test_df=test_df
        )
        
        print(f"✅ Prediction completed!")
        print(f"📊 Results: {result.predictions}")
        
        if hasattr(result, 'metadata') and 'saved_files' in result.metadata:
            print(f"📁 Files saved: {result.metadata['saved_files']}")
        else:
            print("⚠️ No saved files information found in result metadata")
        
        # Check if experiment directory was created
        experiment_info = llm_classifier.results_manager.get_experiment_info()
        print(f"📂 Experiment directory: {experiment_info['experiment_dir']}")
        
        # Check if files exist
        predictions_dir = experiment_info['experiment_dir'] + "/predictions"
        metrics_dir = experiment_info['experiment_dir'] + "/metrics"
        
        if os.path.exists(predictions_dir):
            files = os.listdir(predictions_dir)
            print(f"📄 Prediction files: {files}")
        else:
            print("❌ Predictions directory not found")
            
        if os.path.exists(metrics_dir):
            files = os.listdir(metrics_dir)
            print(f"📈 Metrics files: {files}")
        else:
            print("❌ Metrics directory not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Testing LLM Results Saving Functionality")
    print("=" * 50)
    
    success = test_llm_results_saving()
    
    if success:
        print("\n✅ LLM results saving test completed!")
    else:
        print("\n❌ LLM results saving test failed!")