"""Example script demonstrating the standard ML project structure for saving predictions and artifacts."""

import pandas as pd
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.core.types import ModelConfig, ModelType, EnsembleConfig
from textclassify.utils.results_manager import ResultsManager


def main():
    """Demonstrate the standard project structure for ML results."""
    
    print("🚀 Standard ML Project Structure Demo")
    print("="*50)
    
    # Load sample data
    print("📊 Loading sample datasets...")
    train_df = pd.read_csv("data/ag_news/ag_train_balanced.csv").head(20)  # Small sample
    val_df = pd.read_csv("data/ag_news/ag_val_balanced.csv").head(10)
    test_df = pd.read_csv("data/ag_news/ag_test_balanced.csv").head(5)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # 1. ROBERTA CLASSIFIER WITH AUTOMATIC RESULTS SAVING
    print("\n🤖 1. RoBERTa Classifier with Standard Results Structure")
    print("-"*60)
    
    ml_config = ModelConfig(
        model_name="roberta-base",
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            "model_name": "roberta-base",
            "learning_rate": 2e-5,
            "num_epochs": 1,
            "batch_size": 4,
            "max_length": 128
        }
    )
    
    # Initialize with automatic results saving
    roberta_classifier = RoBERTaClassifier(
        config=ml_config,
        text_column='description',
        label_columns=["label_1", "label_2", "label_3", "label_4"],
        multi_label=False,
        auto_save_results=True,
        output_dir="outputs",
        experiment_name="roberta_ag_news_demo"
    )
    
    # Train and automatically save results
    print("🏋️ Training RoBERTa...")
    training_result = roberta_classifier.fit(train_df, val_df)
    print(f"📁 Results directory: {training_result.get('output_directory', 'Not saved')}")
    
    # Predict and automatically save results
    print("🔮 Making predictions...")
    test_result = roberta_classifier.predict(test_df)
    print(f"✅ Predictions: {test_result.predictions}")
    
    # 2. FUSION ENSEMBLE WITH COMPREHENSIVE RESULTS SAVING
    print("\n🔗 2. Fusion Ensemble with Comprehensive Results Structure")
    print("-"*60)
    
    # Configure LLM model
    llm_config = ModelConfig(
        model_name="gpt-4o-mini",
        model_type=ModelType.LLM,
        parameters={
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_completion_tokens": 50
        }
    )
    
    # Create LLM classifier
    llm_classifier = OpenAIClassifier(
        config=llm_config,
        text_column='description',
        label_columns=["label_1", "label_2", "label_3", "label_4"],
        enable_cache=True,
        cache_dir="outputs/cache/llm",
        multi_label=False
    )
    
    # Configure Fusion Ensemble with automatic results saving
    fusion_config = EnsembleConfig(
        ensemble_method="fusion",
        models=[roberta_classifier, llm_classifier],
        parameters={
            "fusion_hidden_dims": [16, 8],
            "num_epochs": 2,
            "batch_size": 4,
            "classification_type": "multi_class",
            "val_llm_cache_path": "outputs/cache/llm/val_predictions",
            "test_llm_cache_path": "outputs/cache/llm/test_predictions",
            "auto_save_results": True,
            "output_dir": "outputs", 
            "experiment_name": "fusion_ag_news_demo"
        }
    )
    
    # Create and train fusion ensemble
    fusion_ensemble = FusionEnsemble(fusion_config)
    fusion_ensemble.add_ml_model(roberta_classifier)
    fusion_ensemble.add_llm_model(llm_classifier)
    
    print("🔗 Training Fusion Ensemble...")
    fusion_training_result = fusion_ensemble.fit(train_df, val_df)
    print(f"📁 Fusion results directory: {fusion_training_result.get('output_directory', 'Not saved')}")
    
    print("🔮 Making fusion predictions...")
    fusion_test_result = fusion_ensemble.predict(test_df)
    print(f"✅ Fusion predictions: {fusion_test_result.predictions}")
    
    # 3. MANUAL RESULTS MANAGEMENT EXAMPLE
    print("\n📋 3. Manual Results Management Example")
    print("-"*60)
    
    # Create a manual results manager
    manual_results = ResultsManager(
        base_output_dir="outputs",
        experiment_name="manual_demo"
    )
    
    # Save custom results
    custom_predictions = {
        'model_name': 'custom_classifier',
        'predictions': ['label_1', 'label_2', 'label_1'],
        'confidence': [0.95, 0.87, 0.92]
    }
    
    # Create a mock ClassificationResult
    from textclassify.core.types import ClassificationResult, ClassificationType
    mock_result = ClassificationResult(
        predictions=['label_1', 'label_2', 'label_1'],
        model_name='custom_classifier',
        classification_type=ClassificationType.MULTI_CLASS
    )
    
    # Save manually
    saved_files = manual_results.save_predictions(
        mock_result, 
        "custom_test", 
        test_df.head(3)
    )
    print(f"📁 Manually saved files: {saved_files}")
    
    # Save custom metrics
    custom_metrics = {
        'accuracy': 0.95,
        'f1_score': 0.87,
        'precision': 0.92
    }
    
    metrics_file = manual_results.save_metrics(
        custom_metrics,
        "custom_test",
        "custom_classifier"
    )
    print(f"📊 Metrics saved to: {metrics_file}")
    
    # 4. EXPERIMENT OVERVIEW
    print("\n📈 4. Experiment Overview")
    print("-"*60)
    
    # List all experiments
    all_experiments = manual_results.list_experiments()
    print(f"🔬 Total experiments found: {len(all_experiments)}")
    
    for exp in all_experiments[:3]:  # Show first 3
        print(f"  📋 {exp['experiment_id']}")
        print(f"     Path: {exp['path']}")
        print(f"     Created: {exp.get('created_at', 'Unknown')}")
    
    # 5. DIRECTORY STRUCTURE OVERVIEW
    print("\n📁 5. Generated Directory Structure")
    print("-"*60)
    
    def show_directory_structure(path, max_depth=3, current_depth=0):
        """Show directory structure."""
        if current_depth >= max_depth:
            return
        
        try:
            path = Path(path)
            if not path.exists():
                return
            
            items = list(path.iterdir())
            items.sort(key=lambda x: (x.is_file(), x.name))
            
            for item in items[:10]:  # Limit items shown
                indent = "  " * current_depth
                if item.is_dir():
                    print(f"{indent}📁 {item.name}/")
                    show_directory_structure(item, max_depth, current_depth + 1)
                else:
                    size = item.stat().st_size if item.exists() else 0
                    size_str = f"({size} bytes)" if size < 1024 else f"({size//1024}KB)"
                    print(f"{indent}📄 {item.name} {size_str}")
        except (PermissionError, OSError):
            pass
    
    print("Generated outputs structure:")
    show_directory_structure("outputs", max_depth=4)
    
    print("\n✅ Demo completed!")
    print("\n📋 Summary of Standard ML Project Structure:")
    print("""
    outputs/
    ├── experiments/
    │   ├── {timestamp}_{experiment_name}/
    │   │   ├── predictions/
    │   │   │   ├── train_predictions_{hash}.csv
    │   │   │   ├── train_predictions_{hash}.json
    │   │   │   ├── val_predictions_{hash}.csv
    │   │   │   ├── val_predictions_{hash}.json
    │   │   │   ├── test_predictions_{hash}.csv
    │   │   │   └── test_predictions_{hash}.json
    │   │   ├── metrics/
    │   │   │   ├── train_metrics.yaml
    │   │   │   ├── val_metrics.yaml
    │   │   │   └── test_metrics.yaml
    │   │   ├── models/
    │   │   │   └── model_artifacts/
    │   │   ├── logs/
    │   │   │   └── experiment.log
    │   │   ├── plots/
    │   │   ├── roberta_classifier_config.yaml
    │   │   ├── fusion_ensemble_config.yaml
    │   │   └── experiment_summary.yaml
    │   └── ...
    └── cache/
        ├── llm/
        └── datasets/
    
    Key Features:
    ✅ Automatic timestamping and experiment organization
    ✅ Dataset hashing for cache validation
    ✅ Multiple formats (CSV + JSON) for predictions
    ✅ Comprehensive metadata and configuration saving
    ✅ Git commit tracking for reproducibility
    ✅ Automatic directory creation
    ✅ Experiment listing and comparison
    ✅ Model artifact preservation
    ✅ Metrics tracking and visualization ready
    """)


if __name__ == "__main__":
    main()