"""
Automated Fusion Classifier - Simple interface for LLM+ML fusion.

This module provides a single, easy-to-use class that automatically handles:
- ML model creation and training
- LLM model setup
- Fusion ensemble creation and training
- All the complex orchestration behind the scenes

The user only needs to specify which LLM to use in the config.
"""

import pandas as pd
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from ..ml.roberta_classifier import RoBERTaClassifier
from ..llm.deepseek_classifier import DeepSeekClassifier
from ..llm.openai_classifier import OpenAIClassifier
from ..llm.gemini_classifier import GeminiClassifier
from .fusion import FusionEnsemble
from ..config.settings import Config
from ..core.types import ModelType, TrainingData, ModelConfig, ClassificationType, EnsembleConfig, ClassificationResult
from ..core.base import BaseClassifier
from ..core.exceptions import ConfigurationError, ModelTrainingError


class AutoFusionClassifier(BaseClassifier):
    """
    Automated Fusion Classifier that handles LLM+ML fusion automatically.
    
    The user only needs to specify:
    - Which LLM provider to use ('deepseek', 'openai', 'gemini')
    - Basic configuration parameters
    
    Everything else (ML model setup, fusion training, etc.) is handled automatically.
    """
    
    def __init__(self, config: Union[Dict, ModelConfig]):
        """Initialize AutoFusion classifier.
        
        Args:
            config: Configuration dictionary or ModelConfig object
                   Must contain 'llm_provider' key specifying which LLM to use
        
        Example:
            config = {
                'llm_provider': 'deepseek',  # Required: 'deepseek', 'openai', or 'gemini'
                'label_columns': ['positive', 'negative', 'neutral'],  # Required
                'multi_label': False,  # Optional: default False
                'ml_model': 'roberta-base',  # Optional: default 'roberta-base'
                'max_length': 512,  # Optional: default 512
                'batch_size': 16,  # Optional: default 16
                'num_epochs': 3,  # Optional: default 3
                'fusion_epochs': 10,  # Optional: default 10
                'output_dir': 'fusion_output'  # Optional: default 'fusion_output'
            }
        """
        # Convert dict to ModelConfig if needed
        if isinstance(config, dict):
            self.user_config = config
            model_config = ModelConfig(
                model_name=f"auto_fusion_{config.get('llm_provider', 'unknown')}",
                model_type=ModelType.ENSEMBLE,
                parameters=config
            )
        else:
            self.user_config = config.parameters
            model_config = config
        
        super().__init__(model_config)
        
        # Validate required parameters
        if 'llm_provider' not in self.user_config:
            raise ConfigurationError("'llm_provider' must be specified in config")
        
        if 'label_columns' not in self.user_config:
            raise ConfigurationError("'label_columns' must be specified in config")
        
        # Extract configuration
        self.llm_provider = self.user_config['llm_provider']
        self.label_columns = self.user_config['label_columns']
        self.multi_label = self.user_config.get('multi_label', False)
        self.ml_model_name = self.user_config.get('ml_model', 'roberta-base')
        self.output_dir = Path(self.user_config.get('output_dir', 'fusion_output'))
        
        # Set up classification type
        self.classification_type = ClassificationType.MULTI_LABEL if self.multi_label else ClassificationType.MULTI_CLASS
        self.classes_ = self.label_columns
        
        # Internal components (will be created automatically)
        self.ml_model = None
        self.llm_model = None
        self.fusion_ensemble = None
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"🤖 AutoFusion Classifier initialized")
        print(f"   📊 LLM Provider: {self.llm_provider}")
        print(f"   🏷️  Labels: {self.label_columns}")
        print(f"   📝 Classification: {'Multi-label' if self.multi_label else 'Multi-class'}")
        print(f"   🤖 ML Model: {self.ml_model_name}")
    
    def fit(self, training_data: Union[TrainingData, pd.DataFrame, Dict]) -> None:
        """Train the fusion classifier automatically.
        
        Args:
            training_data: Training data in various formats:
                - TrainingData object
                - pandas DataFrame with text and label columns
                - Dict with 'texts' and 'labels' keys
        """
        print("\n🚀 Starting Automatic Fusion Training...")
        print("="*50)
        
        # Step 1: Create ML model (but don't train yet - fusion will handle training)
        print("1️⃣  Setting up ML model...")
        self.ml_model = self._create_ml_model()
        print(f"   🔧 Created {self.ml_model_name} classifier")
        print("   📝 ML training will be handled by fusion ensemble")
        
        # Step 2: Create LLM model
        print("\n2️⃣  Setting up LLM model...")
        self.llm_model = self._create_llm_model()
        print(f"   🧠 Created {self.llm_provider} classifier")
        
        # Step 3: Create and train fusion ensemble
        print("\n3️⃣  Creating fusion ensemble...")
        self.fusion_ensemble = self._create_fusion_ensemble()
        print("   🔗 Fusion ensemble created")
        
        print("\n4️⃣  Training fusion ensemble...")
        print("   This will automatically:")
        print("   - Process and validate training data format")
        print("   - Train the ML model if needed")
        print("   - Generate LLM predictions on training data")
        print("   - Train fusion MLP to combine ML + LLM predictions")
        print("   (This may take a while...)")
        try:
            # Pass raw training data directly to fusion ensemble
            self.fusion_ensemble.fit(training_data)
            print("   ✅ Fusion training completed successfully!")
        except Exception as e:
            raise ModelTrainingError(f"Fusion training failed: {str(e)}", self.config.model_name)
        
        # Step 5: Save everything
        print("\n5️⃣  Saving trained models...")
        self._save_models()
        print(f"   💾 Models saved to {self.output_dir}")
        
        self.is_trained = True
        
        print("\n🎉 AutoFusion training completed successfully!")
        print("="*50)
    
    def predict(self, texts: Union[List[str], pd.DataFrame, str], 
                true_labels: Optional[List[List[int]]] = None) -> ClassificationResult:
        """Make predictions using the trained fusion classifier.
        
        Args:
            texts: Input texts in various formats:
                - List of strings
                - pandas DataFrame with text column
                - Single string
            true_labels: Optional true labels for evaluation (will be added to DataFrame if provided)
            
        Returns:
            ClassificationResult with predictions and optional metrics
        """
        if not self.is_trained:
            raise ModelTrainingError("AutoFusion classifier must be trained before prediction", self.config.model_name)
        
        # Convert all input formats to DataFrame
        if isinstance(texts, str):
            # Single string - convert to DataFrame
            input_df = pd.DataFrame({'text': [texts]})
        elif isinstance(texts, list):
            # List of texts - convert to DataFrame
            input_df = pd.DataFrame({'text': texts})
        elif isinstance(texts, pd.DataFrame):
            # Already a DataFrame - use as is
            input_df = texts.copy()
        else:
            raise ValueError(f"Unsupported input type: {type(texts)}")
        
        # Add true labels to DataFrame if provided
        if true_labels is not None:
            if len(true_labels) != len(input_df):
                raise ValueError(f"Number of true labels ({len(true_labels)}) must match number of texts ({len(input_df)})")
            
            # Add label columns to DataFrame
            for i, label_col in enumerate(self.label_columns):
                if i < len(true_labels[0]):
                    input_df[label_col] = [labels[i] for labels in true_labels]
        
        # Make predictions using fusion ensemble (now only accepts DataFrame)
        result = self.fusion_ensemble.predict(input_df)
        
        # Add AutoFusion metadata
        if result.metadata is None:
            result.metadata = {}
        
        result.metadata.update({
            'auto_fusion': {
                'llm_provider': self.llm_provider,
                'ml_model': self.ml_model_name,
                'classification_type': self.classification_type.value,
                'label_columns': self.label_columns
            }
        })
        
        return result
    
    def predict_single(self, text: str) -> Union[str, List[str]]:
        """Predict a single text (convenience method).
        
        Args:
            text: Single text to classify
            
        Returns:
            Single prediction (string for multi-class, list for multi-label)
        """
        result = self.predict([text])
        return result.predictions[0]
    
    def evaluate(self, test_data: Union[pd.DataFrame, Dict, TrainingData],
                 return_detailed: bool = False) -> Dict[str, Any]:
        """Evaluate the model on test data.
        
        Args:
            test_data: Test data in various formats
            return_detailed: Whether to return detailed evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Prepare test data
        if isinstance(test_data, pd.DataFrame):
            text_column = self._detect_text_column(test_data)
            test_texts = test_data[text_column].tolist()
            test_labels = self._extract_labels_from_df(test_data)
        elif isinstance(test_data, dict):
            test_texts = test_data['texts']
            test_labels = test_data['labels']
        else:  # TrainingData
            test_texts = test_data.texts
            test_labels = test_data.labels
        
        # Make predictions with evaluation
        result = self.predict(test_texts, test_labels)
        
        # Extract metrics
        metrics = result.metadata.get('metrics', {}) if result.metadata else {}
        
        if return_detailed:
            return {
                'metrics': metrics,
                'predictions': result.predictions,
                'true_labels': test_labels,
                'model_info': result.metadata.get('auto_fusion', {})
            }
        else:
            return metrics
    
    def _prepare_training_data(self, training_data: Union[TrainingData, pd.DataFrame, Dict]) -> TrainingData:
        """Prepare training data in the correct format."""
        if isinstance(training_data, TrainingData):
            return training_data
        
        elif isinstance(training_data, pd.DataFrame):
            # Extract from DataFrame
            text_column = self._detect_text_column(training_data)
            texts = training_data[text_column].tolist()
            labels = self._extract_labels_from_df(training_data)
            
        elif isinstance(training_data, dict):
            # Extract from dictionary
            texts = training_data['texts']
            labels = training_data['labels']
            
        else:
            raise ValueError("training_data must be TrainingData, DataFrame, or dict")
        
        return TrainingData(
            texts=texts,
            labels=labels,
            classification_type=self.classification_type
        )
    
    def _detect_text_column(self, df: pd.DataFrame) -> str:
        """Automatically detect the text column in a DataFrame."""
        # Check for common text column names
        common_names = ['text', 'content', 'message', 'review', 'comment', 'description']
        
        for name in common_names:
            if name in df.columns:
                return name
        
        # If not found, use the first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        
        raise ValueError("Could not detect text column in DataFrame")
    
    def _extract_labels_from_df(self, df: pd.DataFrame) -> List[List[int]]:
        """Extract labels from DataFrame in the correct format."""
        if self.multi_label:
            # Multi-label: use binary columns directly
            return df[self.label_columns].values.tolist()
        else:
            # Multi-class: check if it's string labels or binary encoding
            if len(self.label_columns) == 1:
                # Single column with string labels
                label_col = self.label_columns[0]
                if df[label_col].dtype == 'object':
                    # Convert string labels to one-hot
                    unique_labels = sorted(df[label_col].unique())
                    self.classes_ = unique_labels
                    labels = []
                    for label in df[label_col]:
                        one_hot = [0] * len(unique_labels)
                        one_hot[unique_labels.index(label)] = 1
                        labels.append(one_hot)
                    return labels
            
            # Binary encoding
            return df[self.label_columns].values.tolist()
    
    def _create_ml_model(self) -> RoBERTaClassifier:
        """Create and configure the ML model."""
        ml_config = ModelConfig(
            model_name=self.ml_model_name,
            model_type=ModelType.TRADITIONAL_ML,
            parameters={
                'model_name': self.ml_model_name,
                'max_length': self.user_config.get('max_length', 512),
                'learning_rate': self.user_config.get('ml_learning_rate', 2e-5),
                'num_epochs': self.user_config.get('num_epochs', 3),
                'batch_size': self.user_config.get('batch_size', 16),
                'label_columns': self.label_columns
            }
        )
        
        return RoBERTaClassifier(config=ml_config)
    
    def _create_llm_model(self):
        """Create and configure the LLM model."""
        llm_config = Config()
        llm_config.model_type = ModelType.LLM
        llm_config.parameters = {
            'model': self.user_config.get('llm_model', self._get_default_llm_model()),
            'temperature': self.user_config.get('temperature', 0.1),
            'max_completion_tokens': self.user_config.get('max_completion_tokens', 150)
        }
        
        # Add provider-specific parameters
        if self.llm_provider == 'openai':
            return OpenAIClassifier(
                config=llm_config,
                label_columns=self.label_columns,
                multi_label=self.multi_label
            )
        elif self.llm_provider == 'gemini':
            llm_config.parameters.update({
                'top_p': self.user_config.get('top_p', 0.95),
                'top_k': self.user_config.get('top_k', 40)
            })
            return GeminiClassifier(
                config=llm_config,
                label_columns=self.label_columns,
                multi_label=self.multi_label
            )
        else:  # deepseek (default)
            llm_config.parameters.update({
                'top_p': self.user_config.get('top_p', 1.0),
                'frequency_penalty': self.user_config.get('frequency_penalty', 0.0),
                'presence_penalty': self.user_config.get('presence_penalty', 0.0)
            })
            return DeepSeekClassifier(
                config=llm_config,
                label_columns=self.label_columns,
                multi_label=self.multi_label
            )
    
    def _get_default_llm_model(self) -> str:
        """Get default model name for the LLM provider."""
        defaults = {
            'deepseek': 'deepseek-chat',
            'openai': 'gpt-3.5-turbo',
            'gemini': 'gemini-1.5-flash'
        }
        return defaults.get(self.llm_provider, 'deepseek-chat')
    
    def _create_fusion_ensemble(self) -> FusionEnsemble:
        """Create and configure the fusion ensemble."""
        fusion_config = EnsembleConfig(
            ensemble_method='fusion',
            models=[self.ml_model, self.llm_model],
            parameters={
                'fusion_hidden_dims': self.user_config.get('fusion_hidden_dims', [64, 32]),
                'ml_lr': self.user_config.get('fusion_ml_lr', 1e-5),
                'fusion_lr': self.user_config.get('fusion_lr', 1e-3),
                'num_epochs': self.user_config.get('fusion_epochs', 10),
                'batch_size': self.user_config.get('batch_size', 16),
                'classification_type': self.classification_type  # Pass classification type
            }
        )
        
        fusion_ensemble = FusionEnsemble(fusion_config)
        fusion_ensemble.add_ml_model(self.ml_model)
        fusion_ensemble.add_llm_model(self.llm_model)
        
        return fusion_ensemble
    
    def _save_models(self):
        """Save all trained models and configuration."""
        import pickle
        import torch
        import yaml
        
        # Save configuration first
        config_path = self.output_dir / 'auto_fusion_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.user_config, f, default_flow_style=False)
        
        # Save basic classifier state (without PyTorch models)
        classifier_state = {
            'llm_provider': self.llm_provider,
            'label_columns': self.label_columns,
            'multi_label': self.multi_label,
            'ml_model_name': self.ml_model_name,
            'classification_type': self.classification_type,
            'classes_': self.classes_,
            'is_trained': self.is_trained,
            'user_config': self.user_config
        }
        
        state_path = self.output_dir / 'classifier_state.pkl'
        with open(state_path, 'wb') as f:
            pickle.dump(classifier_state, f)
        
        # Save PyTorch models separately using torch.save
        if self.fusion_ensemble and hasattr(self.fusion_ensemble, 'fusion_wrapper') and self.fusion_ensemble.fusion_wrapper:
            fusion_model_path = self.output_dir / 'fusion_model.pt'
            torch.save({
                'fusion_mlp_state_dict': self.fusion_ensemble.fusion_wrapper.fusion_mlp.state_dict(),
                'num_labels': self.fusion_ensemble.num_labels,
                'fusion_hidden_dims': self.fusion_ensemble.fusion_hidden_dims,
                'classification_type': self.fusion_ensemble.classification_type
            }, fusion_model_path)
        
        # Save ML model separately if it has a save method
        if self.ml_model and hasattr(self.ml_model, 'save_model'):
            ml_model_path = self.output_dir / 'ml_model'
            self.ml_model.save_model(str(ml_model_path))
        
        print(f"   💾 Saved classifier state to {state_path}")
        print(f"   💾 Saved configuration to {config_path}")
        if self.fusion_ensemble and hasattr(self.fusion_ensemble, 'fusion_wrapper') and self.fusion_ensemble.fusion_wrapper:
            print(f"   💾 Saved fusion model to {self.output_dir / 'fusion_model.pt'}")
        if self.ml_model and hasattr(self.ml_model, 'save_model'):
            print(f"   💾 Saved ML model to {self.output_dir / 'ml_model'}")
    
    @classmethod
    def load(cls, model_path: str) -> 'AutoFusionClassifier':
        """Load a saved AutoFusion classifier.
        
        Args:
            model_path: Path to saved model directory
            
        Returns:
            Loaded AutoFusionClassifier
        """
        import pickle
        import torch
        
        model_path = Path(model_path)
        
        if not model_path.is_dir():
            raise ValueError(f"Model path must be a directory: {model_path}")
        
        # Load classifier state
        state_path = model_path / 'classifier_state.pkl'
        if not state_path.exists():
            raise FileNotFoundError(f"Classifier state not found: {state_path}")
        
        with open(state_path, 'rb') as f:
            classifier_state = pickle.load(f)
        
        # Recreate the classifier
        classifier = cls(classifier_state['user_config'])
        
        # Restore state
        classifier.llm_provider = classifier_state['llm_provider']
        classifier.label_columns = classifier_state['label_columns']
        classifier.multi_label = classifier_state['multi_label']
        classifier.ml_model_name = classifier_state['ml_model_name']
        classifier.classification_type = classifier_state['classification_type']
        classifier.classes_ = classifier_state['classes_']
        classifier.is_trained = classifier_state['is_trained']
        
        # Recreate the models
        classifier.ml_model = classifier._create_ml_model()
        classifier.llm_model = classifier._create_llm_model()
        classifier.fusion_ensemble = classifier._create_fusion_ensemble()
        
        # Load ML model if it was saved
        ml_model_path = model_path / 'ml_model'
        if ml_model_path.exists() and hasattr(classifier.ml_model, 'load_model'):
            classifier.ml_model.load_model(str(ml_model_path))
            classifier.ml_model.is_trained = True
        
        # Load fusion model if it was saved
        fusion_model_path = model_path / 'fusion_model.pt'
        if fusion_model_path.exists():
            fusion_data = torch.load(fusion_model_path, map_location='cpu')
            
            # Recreate fusion wrapper with correct parameters
            from .fusion import FusionWrapper
            task = "multilabel" if classifier.classification_type == ClassificationType.MULTI_LABEL else "multiclass"
            classifier.fusion_ensemble.fusion_wrapper = FusionWrapper(
                ml_model=classifier.ml_model,
                num_labels=fusion_data['num_labels'],
                task=task,
                hidden_dims=fusion_data['fusion_hidden_dims']
            )
            
            # Load the trained weights
            classifier.fusion_ensemble.fusion_wrapper.fusion_mlp.load_state_dict(fusion_data['fusion_mlp_state_dict'])
            classifier.fusion_ensemble.num_labels = fusion_data['num_labels']
            classifier.fusion_ensemble.fusion_hidden_dims = fusion_data['fusion_hidden_dims']
            classifier.fusion_ensemble.is_trained = True
        
        print(f"✅ Loaded AutoFusion classifier from {model_path}")
        return classifier
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().model_info
        info.update({
            'auto_fusion': {
                'llm_provider': self.llm_provider,
                'ml_model': self.ml_model_name,
                'label_columns': self.label_columns,
                'multi_label': self.multi_label,
                'is_fusion_trained': self.fusion_ensemble.is_trained if self.fusion_ensemble else False,
                'output_dir': str(self.output_dir)
            }
        })
        return info
