"""Fusion ensemble combining ML and LLM classifiers with trainable MLP."""

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

from ..core.types import ClassificationResult, ClassificationType, EnsembleConfig
from ..core.exceptions import EnsembleError, ModelTrainingError
from ..utils.results_manager import ResultsManager, ModelResultsManager
from .base import BaseEnsemble


class FusionMLP(nn.Module):
    """Multi-layer perceptron for fusing ML and LLM predictions.
    
    Learns optimal combination weights for ML and LLM features through supervised training.
    Uses dropout regularization to prevent overfitting on small fusion training sets.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [64, 32]):
        """Initialize Fusion MLP architecture.
        
        Args:
            input_dim: Input feature dimension (768 RoBERTa hidden states + 4 LLM scores = 772)
            output_dim: Number of output classes
            hidden_dims: List of hidden layer sizes for progressive dimensionality reduction
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with ReLU activation and dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)  # Regularization to prevent overfitting
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fusion network.
        
        Args:
            x: Concatenated ML and LLM features
            
        Returns:
            Class logits before softmax
        """
        return self.network(x)


class FusionWrapper(nn.Module):
    """Wrapper that combines ML model hidden states with LLM predictions through trainable fusion.
    
    This module orchestrates the extraction of ML features and their combination with LLM outputs.
    Supports two modes: hidden state fusion (rich 768-dim features) or probability fusion (sparse 4-dim).
    """
    
    def __init__(self, ml_model, num_labels: int, task: str = "multiclass", 
                 hidden_dims: List[int] = [64, 32], use_hidden_states: bool = True):
        """Initialize fusion wrapper with ML model and configuration.
        
        Args:
            ml_model: Pre-trained ML classifier (typically RoBERTa-based)
            num_labels: Number of classification labels
            task: Classification task type ("multiclass" or "multilabel")
            hidden_dims: Hidden layer dimensions for fusion MLP
            use_hidden_states: Whether to use ML hidden states (768-dim) or probabilities (4-dim).
                              Hidden states provide richer semantic representation.
        """
        super().__init__()
        self.ml_model = ml_model
        self.num_labels = num_labels
        self.task = task
        self.use_hidden_states = use_hidden_states
        
        # Calculate fusion MLP input dimension based on feature type
        if use_hidden_states:
            # Rich representation: RoBERTa [CLS] hidden state (768-dim) + LLM predictions (num_labels)
            roberta_hidden_size = 768
            fusion_input_dim = roberta_hidden_size + num_labels
            print(f"   Using RoBERTa hidden states ({roberta_hidden_size}-dim) + LLM probabilities for rich fusion")
        else:
            # Sparse representation: ML probabilities + LLM probabilities
            fusion_input_dim = num_labels * 2
            print(f"   Using probability-only fusion ({fusion_input_dim}-dim)")
        
        # Initialize trainable fusion network
        self.fusion_mlp = FusionMLP(fusion_input_dim, num_labels, hidden_dims)
        
        # Set device and move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, ml_hidden_or_probs: torch.Tensor, llm_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Combine ML and LLM features through fusion MLP.
        
        Args:
            ml_hidden_or_probs: ML features - either hidden states (batch, 768) or probabilities (batch, num_labels)
            llm_predictions: LLM prediction scores (batch, num_labels)
            
        Returns:
            Dictionary with 'fused_logits' key containing classification logits (batch, num_labels)
        """
        # Detach inputs to prevent gradient flow back to pre-trained models
        ml_hidden_or_probs = ml_hidden_or_probs.detach()
        llm_predictions = llm_predictions.detach()
        
        # Concatenate ML and LLM features along feature dimension
        # Shape examples:
        #   Hidden state mode: ml (batch, 768) + llm (batch, 4) -> fusion_input (batch, 772)
        #   Probability mode: ml (batch, 4) + llm (batch, 4) -> fusion_input (batch, 8)
        fusion_input = torch.cat([ml_hidden_or_probs, llm_predictions], dim=1)
        
        # Pass concatenated features through fusion MLP
        fused_logits = self.fusion_mlp(fusion_input)
        
        return {
            "ml_features": ml_hidden_or_probs,
            "llm_predictions": llm_predictions,
            "fused_logits": fused_logits
        }


class FusionEnsemble(BaseEnsemble):
    """Ensemble that combines ML and LLM classifiers through a trainable fusion layer.
    
    The fusion approach extracts rich semantic features from the ML model (RoBERTa hidden states)
    and combines them with LLM predictions using a trainable MLP. This allows the system to learn
    optimal combination weights based on validation data.
    
    Key features:
    - Extracts 768-dimensional hidden states from RoBERTa [CLS] token
    - Combines with LLM predictions (binary vectors or probabilities)
    - Trains a small MLP (64->32->num_classes) to fuse features
    - Implements early stopping to prevent overfitting
    - Supports automatic caching of LLM predictions
    """
    
    def __init__(
        self, 
        ensemble_config,
        # Results management parameters
        output_dir: str = "outputs",
        experiment_name: Optional[str] = None,
        auto_save_results: bool = True,
        save_intermediate_llm_predictions: bool = False,
        # Cache management parameters
        auto_use_cache: bool = False,
        cache_dir: str = "cache"
    ):
        """Initialize fusion ensemble with ML and LLM models.
        
        Args:
            ensemble_config: Ensemble configuration containing fusion parameters
            output_dir: Directory for saving experiment results
            experiment_name: Unique name for this experiment run
            auto_save_results: Whether to automatically save predictions and metrics
            save_intermediate_llm_predictions: Whether to save LLM predictions during training
            auto_use_cache: Whether to automatically load cached LLM predictions if available
            cache_dir: Base directory for caching LLM predictions
            auto_use_cache: Whether to automatically check and reuse cached LLM predictions (default: False)
            cache_dir: Directory to search for cached predictions (default: "cache")
        """
        super().__init__(
            ensemble_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            auto_save_results=False  # Disable base ensemble results manager to prevent duplicate directories
        )
        
        # Fusion-specific parameters
        self.fusion_hidden_dims = ensemble_config.parameters.get('fusion_hidden_dims', [64, 32])
        self.ml_lr = ensemble_config.parameters.get('ml_lr', 1e-5)  # Small LR for ML backbone
        self.fusion_lr = ensemble_config.parameters.get('fusion_lr', 1e-3)  # Larger LR for fusion MLP
        self.num_epochs = ensemble_config.parameters.get('num_epochs', 10)
        self.batch_size = ensemble_config.parameters.get('batch_size', 16)
        self.save_intermediate_llm_predictions = save_intermediate_llm_predictions
        
        # Cache management settings
        self.auto_use_cache = auto_use_cache
        self.cache_dir = cache_dir
        
        # Model components
        self.ml_model = None
        self.llm_model = None
        self.fusion_wrapper = None
        self.llm_scores_cache = {}
        self.calibrator = None
        self.test_performance = {}  # Store test set performance
        
        # LLM prediction cache file paths from ensemble config
        self.val_llm_cache_path = ensemble_config.parameters.get('val_llm_cache_path', '')
        self.test_llm_cache_path = ensemble_config.parameters.get('test_llm_cache_path', '')
        
        # Results management
        output_dir = ensemble_config.parameters.get('output_dir', 'outputs')
        experiment_name = ensemble_config.parameters.get('experiment_name', 'fusion_ensemble')
        auto_save_results = ensemble_config.parameters.get('auto_save_results', True)
        
        self.results_manager = None
        if auto_save_results:
            self.results_manager = ResultsManager(
                base_output_dir=output_dir,
                experiment_name=experiment_name
            )
            self.model_results_manager = ModelResultsManager(
                self.results_manager, 
                f"fusion_ensemble_{self.results_manager.experiment_id}"
            )
        
        # Initialize training state
        self.is_trained = False
        
        # Determine classification type from ensemble config or infer from models
        if 'classification_type' in ensemble_config.parameters:
            config_type = ensemble_config.parameters['classification_type']
            if isinstance(config_type, str):
                # Convert string to enum
                if config_type.lower() in ['multi_label', 'multilabel']:
                    self.classification_type = ClassificationType.MULTI_LABEL
                elif config_type.lower() in ['multi_class', 'multiclass', 'single_label']:
                    self.classification_type = ClassificationType.MULTI_CLASS
                else:
                    self.classification_type = ClassificationType.MULTI_CLASS  # Default
            else:
                self.classification_type = config_type
        elif 'multi_label' in ensemble_config.parameters:
            self.classification_type = ClassificationType.MULTI_LABEL if ensemble_config.parameters['multi_label'] else ClassificationType.MULTI_CLASS
        else:
            # Will be determined when models are added or during fit
            self.classification_type = None
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def add_ml_model(self, ml_model):
        """Add ML model (e.g., RoBERTa) to the fusion ensemble."""
        self.ml_model = ml_model
        self.models.append(ml_model)
        self.model_names.append("ml_model")
    
    def add_llm_model(self, llm_model):
        """Add LLM model to the fusion ensemble.
        
        If auto_use_cache is enabled in the ensemble, it will be propagated to the LLM model.
        """
        self.llm_model = llm_model
        
        # Propagate cache settings to LLM model if supported
        if hasattr(llm_model, 'auto_use_cache') and hasattr(llm_model, 'cache_dir'):
            llm_model.auto_use_cache = self.auto_use_cache
            llm_model.cache_dir = self.cache_dir
            if self.auto_use_cache and hasattr(llm_model, 'verbose') and llm_model.verbose:
                print(f"🔗 FusionEnsemble: Propagated auto_use_cache={self.auto_use_cache} to LLM model")
        
        self.models.append(llm_model)
        self.model_names.append("llm_model")
    
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
            val_llm_predictions: Optional[List[Union[str, List[str]]]] = None) -> Dict[str, Any]:
        """Train the fusion ensemble with train and validation DataFrames.
        
        Args:
            train_df: Training DataFrame with text and label columns
            val_df: Validation DataFrame with text and label columns
            val_llm_predictions: Optional pre-computed LLM predictions for validation set.
                                 If provided, skips LLM inference on validation data.
            
        Returns:
            Dictionary with training results and metrics
        """
        if self.ml_model is None or self.llm_model is None:
            raise EnsembleError("Both ML and LLM models must be added before training")
        
        # Determine classification type if not already set
        if self.classification_type is None:
            if hasattr(self.ml_model, 'multi_label'):
                self.classification_type = ClassificationType.MULTI_LABEL if self.ml_model.multi_label else ClassificationType.MULTI_CLASS
            elif hasattr(self.llm_model, 'multi_label'):
                self.classification_type = ClassificationType.MULTI_LABEL if self.llm_model.multi_label else ClassificationType.MULTI_CLASS
            else:
                # Default to multi-class
                self.classification_type = ClassificationType.MULTI_CLASS
        
        # Extract text and label columns - assume they match ML model configuration
        text_column = self.ml_model.text_column or 'text'
        label_columns = self.ml_model.label_columns or [col for col in train_df.columns if col != text_column]
        
        print(f"Classification type: {self.classification_type}")
        print(f"Training data: {len(train_df)} samples")
        print(f"Validation data: {len(val_df)} samples")
        
        # Step 1: Train ML model on training set
        if not self.ml_model.is_trained:
            print("Training ML model on training set...")
            # Use DataFrame interface directly
            self.ml_model.fit(train_df, val_df)
        else:
            print("ML model already trained")
        
        # Set up classes from ML model
        self.classes_ = self.ml_model.classes_
        self.num_labels = len(self.classes_)
        
        # Step 2: Get ML predictions on validation set
        print("Getting ML predictions on validation set...")
        ml_val_result = self.ml_model.predict_without_saving(val_df)
        
        # Step 3: Get LLM predictions on validation set
        llm_val_predictions = self._get_or_generate_llm_predictions(
            df=val_df,
            train_df=train_df,
            cache_path=self.val_llm_cache_path,
            dataset_type="validation",
            provided_predictions=val_llm_predictions
        )
        
        # Create ClassificationResult for validation predictions
        # Extract true labels from validation DataFrame
        val_true_labels = None
        if all(col in val_df.columns for col in label_columns):
            val_true_labels = val_df[label_columns].values.tolist()
        
        # If we have true labels, call the LLM classifier's predict_texts method to get proper metrics
        if val_true_labels is not None and self.llm_model is not None:
            try:
                # Temporarily disable LLM results saving if intermediate saving is disabled
                original_llm_results_manager = None
                if not self.save_intermediate_llm_predictions and hasattr(self.llm_model, 'results_manager'):
                    original_llm_results_manager = self.llm_model.results_manager
                    self.llm_model.results_manager = None
                
                try:
                    # Call the LLM classifier's predict_texts method with cached predictions
                    # This ensures metrics are calculated and individual results are saved
                    llm_val_result = self.llm_model.predict_texts(
                        texts=val_df[text_column].tolist(), 
                        true_labels=val_true_labels
                    )
                    # Override predictions with cached ones if they were used
                    if val_llm_predictions is not None or self.val_llm_cache_path:
                        llm_val_result.predictions = llm_val_predictions
                        # Update model name to indicate cached predictions were used
                        llm_val_result.model_name = "llm_model_cached"
                finally:
                    # Restore the original results manager
                    if original_llm_results_manager is not None:
                        self.llm_model.results_manager = original_llm_results_manager
            except Exception as e:
                print(f"Warning: Could not call LLM classifier predict_texts for metrics: {e}")
                # Fallback to simple ClassificationResult
                from ..core.types import ClassificationResult
                llm_val_result = ClassificationResult(
                    predictions=llm_val_predictions,
                    model_name="llm_model" if val_llm_predictions is None else "llm_model_cached",
                    classification_type=self.classification_type
                )
        else:
            # No true labels available, create simple ClassificationResult
            from ..core.types import ClassificationResult
            llm_val_result = ClassificationResult(
                predictions=llm_val_predictions,
                model_name="llm_model" if val_llm_predictions is None else "llm_model_cached",
                classification_type=self.classification_type
            )
        
        # Step 4: Create fusion wrapper
        print("Creating fusion wrapper...")
        print("   Using RoBERTa hidden states (768-dim) + LLM probabilities for rich fusion")
        task = "multilabel" if self.classification_type == ClassificationType.MULTI_LABEL else "multiclass"
        self.fusion_wrapper = FusionWrapper(
            ml_model=self.ml_model,
            num_labels=self.num_labels,
            task=task,
            hidden_dims=self.fusion_hidden_dims,
            use_hidden_states=True  # Use RoBERTa hidden states for better fusion
        )
        
        # Step 5: Train fusion MLP on validation set predictions
        print("Training fusion MLP on validation predictions...")
        self._train_fusion_mlp_on_val(val_df, ml_val_result, llm_val_result, text_column, label_columns)
        
        # Step 6: Generate and save fusion predictions on full validation set
        print("Generating fusion predictions on validation set...")
        val_true_labels = val_df[label_columns].values.tolist() if all(col in val_df.columns for col in label_columns) else None
        fusion_val_result = self._predict_with_fusion(
            ml_val_result, 
            llm_val_result, 
            val_df[text_column].tolist(), 
            val_true_labels
        )
        
        # Save fusion validation predictions to experiments directory
        if self.results_manager and fusion_val_result:
            try:
                saved_files = self.results_manager.save_predictions(
                    fusion_val_result, "validation", val_df
                )
                
                # Save metrics if available
                if hasattr(fusion_val_result, 'metadata') and fusion_val_result.metadata and 'metrics' in fusion_val_result.metadata:
                    metrics_file = self.results_manager.save_metrics(
                        fusion_val_result.metadata['metrics'], "validation", "fusion_ensemble"
                    )
                    saved_files["metrics"] = metrics_file
                
                print(f"📁 Validation results saved: {saved_files}")
                
            except Exception as e:
                print(f"Warning: Could not save validation results: {e}")
        
        # Cache training data for later LLM predictions
        self.train_df_cache = train_df.copy()
        
        # Set training flag
        self.is_trained = True
        
        # Return training results (similar to RoBERTa)
        training_result = {
            'ensemble_method': 'fusion',
            'training_samples': len(train_df),
            'validation_samples': len(val_df),
            'num_labels': self.num_labels,
            'classes': self.classes_,
            'classification_type': self.classification_type.value if hasattr(self.classification_type, 'value') else self.classification_type,
            'ml_model_trained': self.ml_model.is_trained,
            'fusion_mlp_trained': True,
            'used_cached_val_llm_predictions': val_llm_predictions is not None
        }
        
        # Save training results using ResultsManager
        if self.results_manager:
            try:
                # Save model configuration
                ensemble_config_dict = {
                    'ensemble_method': 'fusion',
                    'fusion_hidden_dims': self.fusion_hidden_dims,
                    'ml_lr': self.ml_lr,
                    'fusion_lr': self.fusion_lr,
                    'num_epochs': self.num_epochs,
                    'batch_size': self.batch_size,
                    'classification_type': str(self.classification_type),
                    'num_labels': self.num_labels,
                    'classes': self.classes_
                }
                
                self.results_manager.save_model_config(
                    ensemble_config_dict, 
                    "fusion_ensemble"
                )
                
                # Save training summary
                self.results_manager.save_experiment_summary(training_result)
                
                # Get experiment info for user
                exp_info = self.results_manager.get_experiment_info()
                training_result['experiment_info'] = exp_info
                training_result['output_directory'] = exp_info['experiment_dir']
                
                print(f"📁 Results saved to: {exp_info['experiment_dir']}")
                
            except Exception as e:
                print(f"Warning: Could not save training results: {e}")
        
        print("✅ Fusion ensemble training completed!")
        return training_result
    
    def get_llm_predictions(self, df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None) -> List[Union[str, List[str]]]:
        """Get LLM predictions for a DataFrame without training the fusion ensemble.
        
        This is useful for caching LLM predictions that can be reused later.
        
        Args:
            df: DataFrame to get predictions for
            train_df: Optional training DataFrame for few-shot examples.
                     If not provided, uses cached training data or df itself.
            
        Returns:
            List of LLM predictions that can be passed to fit() or predict()
        """
        if self.llm_model is None:
            raise EnsembleError("LLM model must be added before getting predictions")
        
        # Use provided train_df, cached training data, or df itself for few-shot
        if train_df is not None:
            few_shot_df = train_df
        elif hasattr(self, 'train_df_cache') and self.train_df_cache is not None:
            few_shot_df = self.train_df_cache
        else:
            few_shot_df = df
            print("Warning: No training data available for few-shot examples, using target data")
        
        print(f"Getting LLM predictions for {len(df)} samples...")
        llm_result = self.llm_model.predict(
            train_df=few_shot_df,
            test_df=df
        )
        
        return llm_result.predictions
    
    def _get_or_generate_llm_predictions(self, df: pd.DataFrame, train_df: pd.DataFrame, 
                                       cache_path: str, dataset_type: str,
                                       provided_predictions: Optional[List[Union[str, List[str]]]] = None) -> List[Union[str, List[str]]]:
        """Get LLM predictions either from cache, provided predictions, or generate new ones.
        
        For cached/provided predictions, saves metrics directly to experiments folder without
        calling the LLM API. For fresh predictions, uses the normal LLM classifier workflow.
        
        Args:
            df: DataFrame to get predictions for
            train_df: Training DataFrame for few-shot examples
            cache_path: Base path for caching (without datetime extension)
            dataset_type: Type of dataset ("validation" or "test") for logging
            provided_predictions: Optional pre-computed predictions
            
        Returns:
            List of LLM predictions
        """
        final_predictions = None
        predictions_source = "generated"
        
        # STEP 1: Determine prediction source and get predictions
        if provided_predictions is not None:
            print(f"Using provided LLM predictions for {dataset_type} set...")
            final_predictions = provided_predictions
            predictions_source = "provided"
        elif not cache_path or cache_path.strip() == '':
            print(f"No cache path specified, generating LLM predictions for {dataset_type} set on the fly...")
            final_predictions = self._generate_llm_predictions(df, train_df)
            predictions_source = "generated"
        else:
            # Try to load from cache first (with dataset validation)
            cached_predictions = self._load_cached_llm_predictions(cache_path, df)
            if cached_predictions is not None:
                print(f"Loaded cached LLM predictions for {dataset_type} set from {cache_path}")
                final_predictions = cached_predictions
                predictions_source = "cached"
            else:
                # Generate new predictions and save to cache
                print(f"Generating new LLM predictions for {dataset_type} set...")
                final_predictions = self._generate_llm_predictions(df, train_df)
                predictions_source = "generated"
                
                # Save to cache with datetime stamp and dataset hash
                self._save_cached_llm_predictions(final_predictions, cache_path, df)
        
        # STEP 2: Handle results saving based on prediction source
        if self.llm_model is not None and final_predictions is not None:
            try:
                # Extract label columns from DataFrame
                text_column = self.ml_model.text_column if self.ml_model else 'text'
                label_columns = self.ml_model.label_columns if self.ml_model else [col for col in df.columns if col != text_column]
                
                # Check if we have true labels for metrics calculation
                has_true_labels = all(col in df.columns for col in label_columns)
                
                if has_true_labels:
                    # Extract true labels in the format expected by LLM classifier
                    true_labels = df[label_columns].values.tolist()
                    texts = df[text_column].tolist()
                    
                    if predictions_source in ["cached", "provided"]:
                        # For cached/provided predictions: create metrics and save directly WITHOUT calling LLM API
                        print(f"📝 Creating metrics and saving results for {predictions_source} predictions (no API calls)...")
                        
                        # Convert predictions to binary format for metrics calculation
                        binary_predictions = []
                        for pred in final_predictions:
                            if self.llm_model.multi_label:
                                # Multi-label: convert list of class names to binary vector
                                binary_pred = [0] * len(self.llm_model.classes_)
                                if isinstance(pred, list):
                                    for class_name in pred:
                                        if class_name in self.llm_model.classes_:
                                            binary_pred[self.llm_model.classes_.index(class_name)] = 1
                                binary_predictions.append(binary_pred)
                            else:
                                # Single-label: convert class name to binary vector or parse string response
                                if isinstance(pred, str):
                                    # Parse string prediction using LLM model's method
                                    binary_pred = self.llm_model._parse_prediction_response(pred)
                                else:
                                    # Assume it's already a class name
                                    binary_pred = [0] * len(self.llm_model.classes_)
                                    if pred in self.llm_model.classes_:
                                        binary_pred[self.llm_model.classes_.index(pred)] = 1
                                    else:
                                        # Default to first class if prediction not in classes
                                        binary_pred[0] = 1
                                binary_predictions.append(binary_pred)
                        
                        # Calculate metrics
                        metrics = self.llm_model._calculate_metrics(binary_predictions, true_labels)
                        print(f"Calculated metrics for {dataset_type} set: {metrics}")
                        
                        # Create ClassificationResult for saving
                        from ..core.types import ClassificationResult, ClassificationType, ModelType
                        llm_result = ClassificationResult(
                            predictions=final_predictions,
                            probabilities=None,
                            confidence_scores=None,
                            model_name=self.llm_model.config.parameters.get("model", "unknown"),
                            model_type=self.llm_model.config.model_type,
                            classification_type=self.llm_model.config.classification_type if hasattr(self.llm_model.config, 'classification_type') else ClassificationType.MULTI_CLASS,
                            processing_time=0.0,  # No processing time for cached predictions
                            metadata={
                                'metrics': metrics,
                                'total_samples': len(final_predictions),
                                'binary_predictions': binary_predictions,
                                'prediction_source': predictions_source,
                                'dataset_type': dataset_type
                            }
                        )
                        
                        # Save results using the LLM model's results manager
                        if self.llm_model.results_manager:
                            print(f"💾 Saving {predictions_source} prediction results to experiments folder...")
                            
                            # Save predictions using correct ResultsManager methods
                            prediction_files = self.llm_model.results_manager.save_predictions(
                                llm_result, dataset_type, df
                            )
                            
                            # Save metrics separately 
                            metrics_file = self.llm_model.results_manager.save_metrics(
                                metrics, dataset_type, f"{self.llm_model.provider}_classifier"
                            )
                            
                            # Save model configuration
                            config_file = self.llm_model.results_manager.save_model_config(
                                self.llm_model.config.parameters, f"{self.llm_model.provider}_classifier"
                            )
                            
                            # Save experiment summary
                            summary_file = self.llm_model.results_manager.save_experiment_summary({
                                "predictions_source": predictions_source,
                                "dataset_type": dataset_type,
                                "total_samples": len(final_predictions),
                                "metrics": metrics,
                                "model_name": self.llm_model.config.parameters.get("model", "unknown"),
                                "timestamp": pd.Timestamp.now().isoformat()
                            })
                            
                            print(f"✅ {predictions_source.title()} prediction results saved to experiments folder")
                            print(f"📁 Saved files: predictions={prediction_files}, metrics={metrics_file}, config={config_file}")
                        
                        
                    else:
                        # For fresh predictions: use the normal LLM classifier workflow (with API calls)
                        print(f"🔄 Using LLM classifier predict_texts for fresh predictions...")
                        
                        # Temporarily disable LLM results saving if intermediate saving is disabled
                        original_llm_results_manager = None
                        if not self.save_intermediate_llm_predictions and hasattr(self.llm_model, 'results_manager'):
                            original_llm_results_manager = self.llm_model.results_manager
                            self.llm_model.results_manager = None
                        
                        try:
                            llm_result = self.llm_model.predict_texts(
                                texts=texts, 
                                true_labels=true_labels
                            )
                            if not self.save_intermediate_llm_predictions:
                                print(f"✅ Fresh LLM predictions generated (intermediate saving disabled)")
                            else:
                                print(f"✅ Fresh LLM prediction results saved to experiments folder")
                        finally:
                            # Restore the original results manager
                            if original_llm_results_manager is not None:
                                self.llm_model.results_manager = original_llm_results_manager
                    
                else:
                    print(f"ℹ️ No true labels available in DataFrame - skipping metrics calculation")
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not save LLM results for {dataset_type} set: {e}")
                print(f"   Continuing with {predictions_source} predictions only...")
        
        # STEP 3: Save predictions to experiments directory (fallback)
        # Only save as fallback if predictions were cached/provided (not freshly generated)
        # AND if intermediate LLM prediction saving is enabled
        # Fresh predictions already get saved by predict_texts() above
        if predictions_source in ["cached", "provided"] and self.save_intermediate_llm_predictions:
            self._save_llm_predictions_to_experiments(final_predictions, df, dataset_type)
        
        return final_predictions
    
    def _generate_llm_predictions(self, df: pd.DataFrame, train_df: pd.DataFrame) -> List[Union[str, List[str]]]:
        """Generate LLM predictions for a DataFrame."""
        if self.llm_model is None:
            raise EnsembleError("LLM model must be added before generating predictions")
        
        llm_result = self.llm_model.predict(
            train_df=train_df,
            test_df=df
        )
        return llm_result.predictions
    
    def _create_dataset_hash(self, df: pd.DataFrame) -> str:
        """Create a hash of the DataFrame for cache validation.
        
        Args:
            df: DataFrame to hash
            
        Returns:
            8-character hash string
        """
        import hashlib
        
        # Create hash based on DataFrame content
        df_hash = hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()[:8]  # Only first 8 characters
        
        return df_hash
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _load_cached_llm_predictions(self, base_cache_path: str, df: pd.DataFrame) -> Optional[List[Union[str, List[str]]]]:
        """Try to load cached LLM predictions from file with dataset validation.
        
        Args:
            base_cache_path: Base path without datetime extension
            df: DataFrame to validate against
            
        Returns:
            Cached predictions if found and valid, None otherwise
        """
        if not base_cache_path or not base_cache_path.strip():
            return None
        
        try:
            import json
            import os
            import glob
            
            # Calculate current dataset hash
            current_hash = self._create_dataset_hash(df)
            
            # First, try to find files with matching hash
            pattern = f"{base_cache_path}_*_{current_hash}.json"
            matching_files = glob.glob(pattern)
            
            if matching_files:
                # Get the most recent file with matching hash
                latest_file = max(matching_files, key=os.path.getctime)
                
                with open(latest_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Handle both new format (with metadata and IDs) and old format (just predictions)
                if isinstance(cache_data, dict) and 'predictions' in cache_data:
                    predictions = cache_data['predictions']
                    ids = cache_data.get('ids', None)  # Extract IDs if available
                    metadata = cache_data.get('metadata', {})
                    
                    # Validate sample count
                    if metadata.get('num_samples') == len(df):
                        if ids:
                            print(f"✅ Loaded matching cached predictions with IDs: {latest_file} (Hash: {current_hash}, IDs: {len(ids)})")
                        else:
                            print(f"✅ Loaded matching cached predictions: {latest_file} (Hash: {current_hash})")
                        return predictions
                    else:
                        print(f"⚠️ Sample count mismatch in cache: {latest_file}")
                elif isinstance(cache_data, list):
                    # Old format - just validate sample count (no IDs available)
                    if len(cache_data) == len(df):
                        print(f"✅ Loaded compatible cached predictions (legacy format, no IDs): {latest_file} (Hash: {current_hash})")
                        return cache_data
            
            # Fallback: Look for old files without hash (backward compatibility)
            print(f"🔍 No hash-matched cache found, checking backward compatibility...")
            old_pattern = f"{base_cache_path}_*.json"
            old_files = [f for f in glob.glob(old_pattern) if not f.endswith(f"_{current_hash}.json")]
            
            for file_path in sorted(old_files, key=os.path.getctime, reverse=True):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Handle both old and new formats
                    if isinstance(data, list):
                        predictions = data
                        ids = None
                    elif isinstance(data, dict) and 'predictions' in data:
                        predictions = data['predictions']
                        ids = data.get('ids', None)
                    else:
                        continue
                    
                    # Validate sample count
                    if len(predictions) == len(df):
                        if ids:
                            print(f"⚠️ Using backward-compatible cache with IDs: {file_path} (no hash validation, IDs: {len(ids)})")
                        else:
                            print(f"⚠️ Using backward-compatible cache: {file_path} (no hash validation, no IDs)")
                        return predictions
                        
                except Exception:
                    continue
            
            print(f"❌ No compatible cache found for dataset (Hash: {current_hash})")
            return None
            
        except Exception as e:
            print(f"Warning: Could not load cached predictions from {base_cache_path}: {e}")
            return None
    
    def _save_cached_llm_predictions(self, predictions: List[Union[str, List[str]]], base_cache_path: str, df: pd.DataFrame):
        """Save LLM predictions to cache file with datetime stamp and dataset hash.
        
        Args:
            predictions: LLM predictions to save
            base_cache_path: Base path for the cache file
            df: DataFrame for hash calculation and metadata
        """
        if not base_cache_path or not base_cache_path.strip():
            return
        
        try:
            import json
            import os
            from datetime import datetime
            
            # Create directory if it doesn't exist
            cache_dir = os.path.dirname(base_cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            
            # Calculate dataset hash
            dataset_hash = self._create_dataset_hash(df)
            
            # Create filename with datetime stamp and hash
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            cache_filename = f"{base_cache_path}_{timestamp}_{dataset_hash}.json"
            
            # Extract IDs from DataFrame if available (common column names: id, index, row_id, uid)
            ids = None
            for id_col in ['id', 'index', 'row_id', 'uid']:
                if id_col in df.columns:
                    ids = df[id_col].tolist()
                    break
            
            # If no explicit ID column, use DataFrame index
            if ids is None:
                ids = df.index.tolist()
            
            # Create cache data with metadata and IDs
            cache_data = {
                'predictions': predictions,
                'ids': ids,  # Include IDs for mapping back to original data
                'metadata': {
                    'timestamp': timestamp,
                    'num_samples': len(df),
                    'dataset_hash': dataset_hash,
                    'columns': list(df.columns),
                    'created_at': datetime.now().isoformat(),
                    'has_explicit_ids': ids is not None and ids != df.index.tolist()
                }
            }
            
            # Save predictions with metadata and IDs
            with open(cache_filename, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"✅ LLM predictions saved to {cache_filename} (Hash: {dataset_hash}, IDs: {len(ids)})")
            
        except Exception as e:
            print(f"Warning: Could not save predictions to cache {base_cache_path}: {e}")
    
    def _save_llm_predictions_to_experiments(self, predictions: List[Union[str, List[str]]], 
                                           df: pd.DataFrame, dataset_type: str):
        """Save LLM predictions to the experiments directory structure.
        
        Args:
            predictions: LLM predictions to save
            df: DataFrame used for predictions (for metadata)
            dataset_type: Type of dataset ("validation" or "test")
        """
        if not self.results_manager:
            return
            
        try:
            # Create a ClassificationResult for the LLM predictions
            from ..core.types import ClassificationResult
            
            llm_result = ClassificationResult(
                predictions=predictions,
                model_name="llm_model",
                classification_type=self.classification_type,
                metadata={
                    'dataset_type': dataset_type,
                    'num_samples': len(df),
                    'dataset_hash': self._create_dataset_hash(df),
                    'timestamp': self._get_timestamp(),
                    'llm_model_info': {
                        'type': type(self.llm_model).__name__ if self.llm_model else 'unknown',
                        'model_name': getattr(self.llm_model, 'model_name', 'unknown')
                    }
                }
            )
            
            # Save LLM predictions using ResultsManager
            saved_files = self.results_manager.save_predictions(
                llm_result, f"llm_{dataset_type}", df
            )
            
            print(f"📁 LLM {dataset_type} predictions saved to experiments: {saved_files}")
            
        except Exception as e:
            print(f"Warning: Could not save LLM predictions to experiments: {e}")
    
    def save_llm_predictions(self, predictions: List[Union[str, List[str]]], filepath: str):
        """Save LLM predictions to a file for later reuse.
        
        Args:
            predictions: LLM predictions to save
            filepath: Path to save the predictions (JSON format)
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"LLM predictions saved to {filepath}")
    
    def load_llm_predictions(self, filepath: str) -> List[Union[str, List[str]]]:
        """Load LLM predictions from a file.
        
        Args:
            filepath: Path to load predictions from (JSON format)
            
        Returns:
            List of LLM predictions
        """
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both old format (just list) and new format (dict with predictions and IDs)
        if isinstance(data, list):
            predictions = data
            print(f"LLM predictions loaded from {filepath} (legacy format, no IDs)")
        elif isinstance(data, dict) and 'predictions' in data:
            predictions = data['predictions']
            ids = data.get('ids', None)
            if ids:
                print(f"LLM predictions loaded from {filepath} (with {len(ids)} IDs)")
            else:
                print(f"LLM predictions loaded from {filepath} (no IDs)")
        else:
            raise ValueError(f"Invalid cache file format: {filepath}")
        
        return predictions
    
    def load_llm_predictions_with_ids(self, filepath: str) -> Tuple[List[Union[str, List[str]]], Optional[List[Any]]]:
        """Load LLM predictions and IDs from a file.
        
        Args:
            filepath: Path to load predictions from (JSON format)
            
        Returns:
            Tuple of (predictions, ids) where ids may be None for legacy format
        """
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both old format (just list) and new format (dict with predictions and IDs)
        if isinstance(data, list):
            predictions = data
            ids = None
            print(f"LLM predictions loaded from {filepath} (legacy format, no IDs)")
        elif isinstance(data, dict) and 'predictions' in data:
            predictions = data['predictions']
            ids = data.get('ids', None)
            if ids:
                print(f"✅ LLM predictions with IDs loaded from {filepath} ({len(predictions)} predictions, {len(ids)} IDs)")
            else:
                print(f"⚠️ LLM predictions loaded from {filepath} (no IDs available)")
        else:
            raise ValueError(f"Invalid cache file format: {filepath}")
        
        return predictions, ids
    
    def load_cached_predictions_for_dataset(self, df: pd.DataFrame, dataset_type: str = "auto") -> Optional[List[Union[str, List[str]]]]:
        """Load cached LLM predictions for a specific dataset (validation or test).
        
        This function provides a convenient way to load cached LLM predictions without
        having to train the fusion ensemble. It automatically determines which cache
        to use based on dataset_type or attempts to match the dataset.
        
        Args:
            df: DataFrame to load predictions for
            dataset_type: Type of dataset - "validation", "test", or "auto" (default)
                         If "auto", tries both validation and test caches
            
        Returns:
            List of cached LLM predictions if found, None otherwise
        """
        if dataset_type == "validation" or dataset_type == "auto":
            if self.val_llm_cache_path:
                cached_predictions = self._load_cached_llm_predictions(self.val_llm_cache_path, df)
                if cached_predictions is not None:
                    print(f"✅ Found cached validation predictions for {len(df)} samples")
                    return cached_predictions
                elif dataset_type == "validation":
                    print(f"❌ No cached validation predictions found for {len(df)} samples")
                    return None
        
        if dataset_type == "test" or dataset_type == "auto":
            if self.test_llm_cache_path:
                cached_predictions = self._load_cached_llm_predictions(self.test_llm_cache_path, df)
                if cached_predictions is not None:
                    print(f"✅ Found cached test predictions for {len(df)} samples")
                    return cached_predictions
                elif dataset_type == "test":
                    print(f"❌ No cached test predictions found for {len(df)} samples")
                    return None
        
        if dataset_type == "auto":
            print(f"❌ No cached predictions found for {len(df)} samples in either validation or test cache")
        
        return None
    
    def get_cached_predictions_summary(self) -> Dict[str, Any]:
        """Get a summary of available cached predictions.
        
        Returns:
            Dictionary with information about cached predictions
        """
        import glob
        import os
        from pathlib import Path
        
        summary = {
            "validation_cache": {
                "path": self.val_llm_cache_path,
                "files": [],
                "latest_file": None,
                "available": False
            },
            "test_cache": {
                "path": self.test_llm_cache_path,
                "files": [],
                "latest_file": None,
                "available": False
            }
        }
        
        # Check validation cache
        if self.val_llm_cache_path and self.val_llm_cache_path.strip():
            pattern = f"{self.val_llm_cache_path}_*.json"
            files = glob.glob(pattern)
            if files:
                summary["validation_cache"]["files"] = files
                summary["validation_cache"]["latest_file"] = max(files, key=os.path.getctime)
                summary["validation_cache"]["available"] = True
        
        # Check test cache
        if self.test_llm_cache_path and self.test_llm_cache_path.strip():
            pattern = f"{self.test_llm_cache_path}_*.json"
            files = glob.glob(pattern)
            if files:
                summary["test_cache"]["files"] = files
                summary["test_cache"]["latest_file"] = max(files, key=os.path.getctime)
                summary["test_cache"]["available"] = True
        
        return summary
    
    def fit_with_cached_predictions(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                   val_llm_predictions: Optional[List[Union[str, List[str]]]] = None,
                                   force_load_from_cache: bool = False) -> Dict[str, Any]:
        """Train fusion ensemble with automatic cache loading for validation predictions.
        
        This is a convenience method that automatically tries to load cached LLM predictions
        for the validation set before training, reducing the need to regenerate them.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            val_llm_predictions: Optional pre-computed validation predictions.
                                If None, will try to load from cache first.
            force_load_from_cache: If True, only use cached predictions and fail if not found
            
        Returns:
            Training results dictionary
        """
        # Try to load cached validation predictions if not provided
        if val_llm_predictions is None:
            print("🔍 Attempting to load cached validation LLM predictions...")
            cached_val_predictions = self.load_cached_predictions_for_dataset(val_df, "validation")
            
            if cached_val_predictions is not None:
                val_llm_predictions = cached_val_predictions
                print(f"✅ Using cached validation predictions ({len(cached_val_predictions)} samples)")
            elif force_load_from_cache:
                raise EnsembleError(
                    "force_load_from_cache=True but no cached validation predictions found",
                    "FusionEnsemble"
                )
            else:
                print("⚠️ No cached validation predictions found, will generate new ones during training")
        
        # Call the regular fit method
        return self.fit(train_df, val_df, val_llm_predictions)
    
    def predict_with_cached_predictions(self, test_df: pd.DataFrame, 
                                       true_labels: Optional[List[List[int]]] = None,
                                       test_llm_predictions: Optional[List[Union[str, List[str]]]] = None,
                                       force_load_from_cache: bool = False) -> ClassificationResult:
        """Make predictions with automatic cache loading for test predictions.
        
        This is a convenience method that automatically tries to load cached LLM predictions
        for the test set before making predictions.
        
        Args:
            test_df: Test DataFrame
            true_labels: Optional true labels for evaluation
            test_llm_predictions: Optional pre-computed test predictions.
                                 If None, will try to load from cache first.
            force_load_from_cache: If True, only use cached predictions and fail if not found
            
        Returns:
            ClassificationResult with predictions and metrics
        """
        # Try to load cached test predictions if not provided
        if test_llm_predictions is None:
            print("🔍 Attempting to load cached test LLM predictions...")
            cached_test_predictions = self.load_cached_predictions_for_dataset(test_df, "test")
            
            if cached_test_predictions is not None:
                test_llm_predictions = cached_test_predictions
                print(f"✅ Using cached test predictions ({len(cached_test_predictions)} samples)")
            elif force_load_from_cache:
                raise EnsembleError(
                    "force_load_from_cache=True but no cached test predictions found",
                    "FusionEnsemble"
                )
            else:
                print("⚠️ No cached test predictions found, will generate new ones during prediction")
        
        # Call the regular predict method
        return self.predict(test_df, true_labels, test_llm_predictions)
    
    @classmethod
    def discover_cached_predictions(cls, cache_directory: str) -> Dict[str, List[str]]:
        """Discover all cached LLM prediction files in a directory.
        
        This is a utility function to help find cached prediction files that can be used
        for training or prediction without regenerating LLM outputs.
        
        Args:
            cache_directory: Directory to search for cached prediction files
            
        Returns:
            Dictionary mapping cache file patterns to list of matching files
        """
        import glob
        import os
        from pathlib import Path
        
        cache_dir = Path(cache_directory)
        if not cache_dir.exists():
            print(f"Cache directory does not exist: {cache_directory}")
            return {}
        
        # Look for all JSON files that match the cache pattern
        pattern = str(cache_dir / "*_*.json")
        all_files = glob.glob(pattern)
        
        # Group files by base name (without timestamp and hash)
        grouped_files = {}
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            # Extract base name by removing timestamp and hash parts
            parts = file_name.split('_')
            if len(parts) >= 3:  # base_timestamp_hash.json
                base_name = '_'.join(parts[:-2]) if len(parts) > 3 else parts[0]
                if base_name not in grouped_files:
                    grouped_files[base_name] = []
                grouped_files[base_name].append(file_path)
        
        # Sort files within each group by creation time (newest first)
        for base_name in grouped_files:
            grouped_files[base_name].sort(key=os.path.getctime, reverse=True)
        
        return grouped_files
    
    def print_cache_status(self):
        """Print a detailed status of cached predictions for this fusion ensemble."""
        import os
        
        print("\n" + "="*60)
        print("🗂️  FUSION ENSEMBLE CACHE STATUS")
        print("="*60)
        
        summary = self.get_cached_predictions_summary()
        
        # Validation cache status
        val_cache = summary["validation_cache"]
        print(f"\n📊 VALIDATION CACHE:")
        print(f"   Path: {val_cache['path'] or 'Not configured'}")
        if val_cache["available"]:
            print(f"   Status: ✅ Available ({len(val_cache['files'])} files)")
            print(f"   Latest: {os.path.basename(val_cache['latest_file'])}")
        else:
            print(f"   Status: ❌ No cached files found")
        
        # Test cache status
        test_cache = summary["test_cache"]
        print(f"\n🧪 TEST CACHE:")
        print(f"   Path: {test_cache['path'] or 'Not configured'}")
        if test_cache["available"]:
            print(f"   Status: ✅ Available ({len(test_cache['files'])} files)")
            print(f"   Latest: {os.path.basename(test_cache['latest_file'])}")
        else:
            print(f"   Status: ❌ No cached files found")
        
        print("\n💡 USAGE TIPS:")
        print("   • Use fit_with_cached_predictions() to automatically load cached validation predictions")
        print("   • Use predict_with_cached_predictions() to automatically load cached test predictions")
        print("   • Use load_cached_predictions_for_dataset() for manual cache loading")
        print("   • Set force_load_from_cache=True to ensure cached predictions are used")
        print("="*60)
    
    def train_fusion_mlp_on_val(self, val_df: pd.DataFrame, ml_val_result, llm_val_result, 
                                text_column: str, label_columns: List[str]):
        """Public method to train the fusion MLP using validation set predictions from both ML and LLM models."""
        return self._train_fusion_mlp_on_val(val_df, ml_val_result, llm_val_result, text_column, label_columns)

    def _train_fusion_mlp_on_val(self, val_df: pd.DataFrame, ml_val_result, llm_val_result, 
                                text_column: str, label_columns: List[str]):
        """Train the fusion MLP using validation set predictions from both ML and LLM models."""
        
        # Split validation DataFrame into train/val for fusion MLP training
        fusion_train_df, fusion_val_df = train_test_split(
            val_df,
            test_size=0.1, random_state=42, stratify=None
        )
        
        # Split both ML and LLM results accordingly
        split_idx = len(fusion_train_df)
        
        # Create split ClassificationResults for train and validation
        from ..core.types import ClassificationResult
        
        # Split ML results
        fusion_train_ml_result = ClassificationResult(
            predictions=ml_val_result.predictions[:split_idx],
            probabilities=ml_val_result.probabilities[:split_idx] if ml_val_result.probabilities else None,
            model_name=ml_val_result.model_name,
            classification_type=ml_val_result.classification_type
        )
        fusion_val_ml_result = ClassificationResult(
            predictions=ml_val_result.predictions[split_idx:],
            probabilities=ml_val_result.probabilities[split_idx:] if ml_val_result.probabilities else None,
            model_name=ml_val_result.model_name,
            classification_type=ml_val_result.classification_type
        )
        
        # Split LLM results
        fusion_train_llm_result = ClassificationResult(
            predictions=llm_val_result.predictions[:split_idx],
            probabilities=llm_val_result.probabilities[:split_idx] if llm_val_result.probabilities else None,
            model_name=llm_val_result.model_name,
            classification_type=llm_val_result.classification_type
        )
        fusion_val_llm_result = ClassificationResult(
            predictions=llm_val_result.predictions[split_idx:],
            probabilities=llm_val_result.probabilities[split_idx:] if llm_val_result.probabilities else None,
            model_name=llm_val_result.model_name,
            classification_type=llm_val_result.classification_type
        )
        
        print(f"   Fusion training split: {len(fusion_train_df)} samples")
        print(f"   Fusion validation split: {len(fusion_val_df)} samples")
        
        # Create PyTorch datasets for fusion training
        # Datasets extract RoBERTa hidden states on-the-fly and combine with LLM predictions
        train_dataset = self._create_fusion_dataset(
            fusion_train_df[text_column].tolist(), 
            fusion_train_df[label_columns].values.tolist(), 
            fusion_train_ml_result,
            fusion_train_llm_result
        )
        val_dataset = self._create_fusion_dataset(
            fusion_val_df[text_column].tolist(), 
            fusion_val_df[label_columns].values.tolist(), 
            fusion_val_ml_result,
            fusion_val_llm_result
        )
        
        # Create data loaders with shuffling for training, sequential for validation
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Freeze pre-trained ML model parameters to prevent catastrophic forgetting
        # Only the fusion MLP parameters will be optimized
        for param in self.fusion_wrapper.ml_model.model.parameters():
            param.requires_grad = False
        
        # Initialize optimizer for fusion MLP parameters only
        fusion_params = list(self.fusion_wrapper.fusion_mlp.parameters())
        optimizer = torch.optim.AdamW(fusion_params, lr=self.fusion_lr)
        
        # Select loss function based on classification type
        if self.classification_type == ClassificationType.MULTI_CLASS:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Initialize training with early stopping to prevent overfitting
        self.fusion_wrapper.train()
        best_val_loss = float('inf')
        best_model_state = None
        patience = 3  # Number of epochs without improvement before stopping
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Training phase: optimize fusion MLP weights
            total_train_loss = 0
            for batch in train_loader:
                ml_predictions, llm_predictions, labels = batch
                ml_predictions = ml_predictions.to(self.device)
                llm_predictions = llm_predictions.to(self.device)
                labels = labels.to(self.device)
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass through fusion wrapper
                outputs = self.fusion_wrapper(ml_predictions, llm_predictions)
                fused_logits = outputs['fused_logits']
                
                # Convert labels to class indices for multi-class classification
                if self.classification_type == ClassificationType.MULTI_CLASS:
                    labels = torch.argmax(labels, dim=1)
                
                # Compute loss and backpropagate
                loss = criterion(fused_logits, labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            # Validation phase: evaluate without updating weights
            self.fusion_wrapper.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    ml_predictions, llm_predictions, labels = batch
                    ml_predictions = ml_predictions.to(self.device)
                    llm_predictions = llm_predictions.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass only
                    outputs = self.fusion_wrapper(ml_predictions, llm_predictions)
                    fused_logits = outputs['fused_logits']
                    
                    # Convert labels for loss computation
                    if self.classification_type == ClassificationType.MULTI_CLASS:
                        labels = torch.argmax(labels, dim=1)
                    
                    # Compute validation loss
                    loss = criterion(fused_logits, labels)
                    total_val_loss += loss.item()
            
            # Calculate average losses for this epoch
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f"   Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping: check if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save best model weights for later restoration
                best_model_state = {
                    'fusion_mlp': self.fusion_wrapper.fusion_mlp.state_dict(),
                    'epoch': epoch + 1
                }
                patience_counter = 0
                print(f"   New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"   No improvement for {patience_counter} epoch(s)")
                
                # Stop training if no improvement for patience epochs
                if patience_counter >= patience:
                    print(f"   Early stopping triggered at epoch {epoch + 1}")
                    print(f"   Best validation loss: {best_val_loss:.4f} at epoch {best_model_state['epoch']}")
                    break
            
            # Return to training mode for next epoch
            self.fusion_wrapper.train()
        
        # Restore the best model weights after training completes
        if best_model_state is not None:
            self.fusion_wrapper.fusion_mlp.load_state_dict(best_model_state['fusion_mlp'])
            print(f"   Restored best model from epoch {best_model_state['epoch']}")
        
        # Set to evaluation mode for inference
        self.fusion_wrapper.eval()
    
    def _create_fusion_dataset(self, texts: List[str], labels: List[List[int]], 
                              ml_result, llm_result):
        """Create custom dataset that extracts RoBERTa hidden states on-the-fly during training.
        
        This method creates a PyTorch Dataset that combines:
        1. RoBERTa hidden states extracted from [CLS] token (768-dim)
        2. LLM predictions converted to vectors (4-dim binary or probability)
        3. True labels for supervised learning
        
        Args:
            texts: Input texts for hidden state extraction
            labels: Ground truth labels as binary vectors
            ml_result: Classification results from ML model (contains model reference)
            llm_result: Classification results from LLM model (contains predictions)
            
        Returns:
            PyTorch Dataset yielding (ml_hidden_states, llm_vector, labels) tuples
        """
        
        # Convert LLM predictions to tensor representation
        # Two modes: probability distributions (soft) or binary vectors (hard)
        llm_tensor = torch.zeros(len(texts), self.num_labels)
        if hasattr(llm_result, 'probabilities') and llm_result.probabilities:
            # Mode 1: Use probability distributions from LLM if available
            print(f"   Using LLM probability distributions (soft predictions)")
            for i, prob_dict in enumerate(llm_result.probabilities):
                for class_name in self.classes_:
                    class_idx = self.classes_.index(class_name)
                    llm_tensor[i, class_idx] = prob_dict.get(class_name, 0.0)
            # Display example for verification
            if len(llm_result.probabilities) > 0:
                print(f"   Example LLM probabilities: {llm_result.probabilities[0]}")
        else:
            # Mode 2: Convert hard predictions to binary one-hot vectors
            print(f"   LLM has no probabilities - using binary vectors (hard predictions)")
            print(f"   Example: 'label_2' -> [0, 1, 0, 0]")
            for i, prediction in enumerate(llm_result.predictions):
                # Handle string predictions (e.g., "label_2")
                if isinstance(prediction, str):
                    if prediction in self.classes_:
                        class_idx = self.classes_.index(prediction)
                        llm_tensor[i, class_idx] = 1.0
                # Handle list predictions (multi-label or binary vectors)
                elif isinstance(prediction, list):
                    if len(prediction) == self.num_labels and all(isinstance(x, (int, float)) for x in prediction):
                        # Already in binary vector format
                        llm_tensor[i] = torch.tensor(prediction, dtype=torch.float)
                    else:
                        # List of class names - convert to multi-hot encoding
                        for pred_class in prediction:
                            if pred_class in self.classes_:
                                class_idx = self.classes_.index(pred_class)
                                llm_tensor[i, class_idx] = 1.0
        
        # Convert ground truth labels to tensor
        labels_tensor = torch.FloatTensor(labels)
        
        # Define custom dataset class for on-the-fly hidden state extraction
        class FusionDatasetWithHiddenStates(torch.utils.data.Dataset):
            """Dataset that extracts RoBERTa hidden states during training.
            
            This approach avoids storing all hidden states in memory by computing them
            on-demand for each batch. The ML model is frozen, so no gradient computation needed.
            """
            
            def __init__(self, texts, llm_probs, labels, ml_model, tokenizer, max_length, device):
                self.texts = texts
                self.llm_probs = llm_probs
                self.labels = labels
                self.ml_model = ml_model
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.device = device
                
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                """Extract features for a single sample.
                
                Returns:
                    tuple: (hidden_state, llm_prediction, label)
                        - hidden_state: RoBERTa [CLS] token embedding (768-dim)
                        - llm_prediction: LLM prediction vector (4-dim)
                        - label: Ground truth label vector
                """
                # Tokenize input text
                text = str(self.texts[idx])
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Extract hidden states from frozen RoBERTa model
                with torch.no_grad():
                    outputs = self.ml_model.model(
                        input_ids=encoding['input_ids'].to(self.device),
                        attention_mask=encoding['attention_mask'].to(self.device),
                        output_hidden_states=True  # Request hidden states
                    )
                    # Extract [CLS] token from last hidden layer (position 0)
                    # Shape: (1, seq_len, 768) -> (768,)
                    hidden_state = outputs.hidden_states[-1][:, 0, :].squeeze(0)
                
                return hidden_state.cpu(), self.llm_probs[idx], self.labels[idx]
        
        return FusionDatasetWithHiddenStates(
            texts=texts,
            llm_probs=llm_tensor,
            labels=labels_tensor,
            ml_model=self.ml_model,
            tokenizer=self.ml_model.tokenizer,
            max_length=self.ml_model.max_length,
            device=self.device
        )
    
    def predict(self, test_df: pd.DataFrame, true_labels: Optional[List[List[int]]] = None,
                test_llm_predictions: Optional[List[Union[str, List[str]]]] = None) -> ClassificationResult:
        """Predict using fusion ensemble on test DataFrame.
        
        Args:
            test_df: Test DataFrame with text and optionally label columns
            true_labels: Optional true labels for evaluation (deprecated, will be extracted from DataFrame)
            test_llm_predictions: Optional pre-computed LLM predictions for test set.
                                  If provided, skips LLM inference on test data.
            
        Returns:
            ClassificationResult with predictions and metrics
        """
        if not self.is_trained:
            raise EnsembleError("Fusion ensemble must be trained before prediction")
        
        # Extract text and label columns - assume they match ML model configuration
        text_column = self.ml_model.text_column or 'text'
        label_columns = self.ml_model.label_columns or [col for col in test_df.columns if col != text_column]
        
        texts = test_df[text_column].tolist()
        
        # Extract true labels from DataFrame if available
        extracted_labels = None
        if all(col in test_df.columns for col in label_columns):
            extracted_labels = test_df[label_columns].values.tolist()
        elif true_labels is not None:
            extracted_labels = true_labels
        
        # Step 1: Get ML predictions on test data
        print("Getting ML predictions on test data...")
        ml_test_result = self.ml_model.predict(test_df)
        
        # Step 2: Get LLM predictions on test data
        llm_test_predictions = self._get_or_generate_llm_predictions(
            df=test_df,
            train_df=getattr(self, 'train_df_cache', test_df),
            cache_path=self.test_llm_cache_path,
            dataset_type="test",
            provided_predictions=test_llm_predictions
        )
        
        # Create ClassificationResult for test predictions
        # If we have true labels, call the LLM classifier's predict_texts method to get proper metrics
        if extracted_labels is not None and self.llm_model is not None:
            try:
                # Temporarily disable LLM results saving if intermediate saving is disabled
                original_llm_results_manager = None
                if not self.save_intermediate_llm_predictions and hasattr(self.llm_model, 'results_manager'):
                    original_llm_results_manager = self.llm_model.results_manager
                    self.llm_model.results_manager = None
                
                try:
                    # Call the LLM classifier's predict_texts method with cached predictions
                    # This ensures metrics are calculated and individual results are saved
                    llm_test_result = self.llm_model.predict_texts(
                        texts=texts, 
                        true_labels=extracted_labels
                    )
                    # Override predictions with cached ones if they were used
                    if test_llm_predictions is not None or self.test_llm_cache_path:
                        llm_test_result.predictions = llm_test_predictions
                        # Update model name to indicate cached predictions were used
                        llm_test_result.model_name = "llm_model_cached"
                finally:
                    # Restore the original results manager
                    if original_llm_results_manager is not None:
                        self.llm_model.results_manager = original_llm_results_manager
            except Exception as e:
                print(f"Warning: Could not call LLM classifier predict_texts for metrics: {e}")
                # Fallback to simple ClassificationResult
                from ..core.types import ClassificationResult
                llm_test_result = ClassificationResult(
                    predictions=llm_test_predictions,
                    model_name="llm_model" if test_llm_predictions is None else "llm_model_cached",
                    classification_type=self.classification_type
                )
        else:
            # No true labels available, create simple ClassificationResult
            from ..core.types import ClassificationResult
            llm_test_result = ClassificationResult(
                predictions=llm_test_predictions,
                model_name="llm_model" if test_llm_predictions is None else "llm_model_cached",
                classification_type=self.classification_type
            )
        
        # Step 3: Use fusion MLP to combine predictions
        print("Generating fusion predictions...")
        result = self._predict_with_fusion(ml_test_result, llm_test_result, texts, extracted_labels)
        
        # Save test results using ResultsManager
        if self.results_manager:
            try:
                saved_files = self.results_manager.save_predictions(
                    result, "test", test_df
                )
                
                # Save metrics if available
                if hasattr(result, 'metadata') and result.metadata and 'metrics' in result.metadata:
                    metrics_file = self.results_manager.save_metrics(
                        result.metadata['metrics'], "test", "fusion_ensemble"
                    )
                    saved_files["metrics"] = metrics_file
                
                print(f"📁 Test results saved: {saved_files}")
                
                # Add file paths to result metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata['saved_files'] = saved_files
                
            except Exception as e:
                print(f"Warning: Could not save test results: {e}")
        
        return result
    
    def _predict_with_fusion(self, ml_result, llm_result, texts: List[str], true_labels: Optional[List[List[int]]] = None) -> ClassificationResult:
        """Generate fusion predictions using trained MLP."""
        # Create dataset using both ML and LLM ClassificationResults (with probabilities!)
        dummy_labels = [[0] * self.num_labels] * len(texts)
        dataset = self._create_fusion_dataset(texts, dummy_labels, ml_result, llm_result)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Generate predictions
        self.fusion_wrapper.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                ml_predictions, llm_predictions, _ = batch
                ml_predictions = ml_predictions.to(self.device)
                llm_predictions = llm_predictions.to(self.device)
                
                outputs = self.fusion_wrapper(ml_predictions, llm_predictions)
                fused_logits = outputs['fused_logits']
                
                if self.classification_type == ClassificationType.MULTI_CLASS:
                    predictions = torch.argmax(fused_logits, dim=-1)
                    for pred_idx in predictions.cpu().numpy():
                        all_predictions.append(self.classes_[pred_idx])
                else:
                    probabilities = torch.sigmoid(fused_logits)
                    threshold = 0.5
                    predictions = (probabilities > threshold).cpu().numpy()
                    for pred_array in predictions:
                        active_labels = [self.classes_[i] for i, is_active in enumerate(pred_array) if is_active]
                        all_predictions.append(active_labels)
        
        return self._create_result(predictions=all_predictions, true_labels=true_labels)
    
    def _combine_predictions(self, model_results: List[ClassificationResult], texts: List[str]) -> List[Union[str, List[str]]]:
        """Not used in fusion ensemble - predictions are generated directly."""
        raise NotImplementedError("Fusion ensemble generates predictions directly")
    
    def _combine_predictions_with_probabilities(self, model_results: List[ClassificationResult], texts: List[str]) -> tuple:
        """Not used in fusion ensemble - predictions are generated directly."""
        raise NotImplementedError("Fusion ensemble generates predictions directly")

    def _create_result(
        self,
        predictions: List[Union[str, List[str]]],
        probabilities: Optional[List[Dict[str, float]]] = None,
        confidence_scores: Optional[List[float]] = None,
        true_labels: Optional[List[List[int]]] = None
    ) -> ClassificationResult:
        """Create ClassificationResult with metrics calculation if true labels provided."""
        
        # Convert predictions to binary vector format for metrics calculation
        binary_predictions = []
        for pred in predictions:
            if isinstance(pred, str):
                # Single-label: convert class name to binary vector
                binary_vector = [0] * self.num_labels
                if pred in self.classes_:
                    class_idx = self.classes_.index(pred)
                    binary_vector[class_idx] = 1
                binary_predictions.append(binary_vector)
            elif isinstance(pred, list) and all(isinstance(x, str) for x in pred):
                # Multi-label: convert class names to binary vector
                binary_vector = [0] * self.num_labels
                for class_name in pred:
                    if class_name in self.classes_:
                        class_idx = self.classes_.index(class_name)
                        binary_vector[class_idx] = 1
                binary_predictions.append(binary_vector)
            elif isinstance(pred, list) and all(isinstance(x, (int, float)) for x in pred):
                # Already in binary vector format
                binary_predictions.append([int(x) for x in pred])
            else:
                # Fallback: create zero vector
                binary_predictions.append([0] * self.num_labels)
        
        # Calculate metrics if true labels are provided
        metrics = None
        if true_labels is not None:
            metrics = self._calculate_metrics(binary_predictions, true_labels)
        
        # Create base result using inherited method
        result = super()._create_result(
            predictions=predictions,
            probabilities=probabilities,
            confidence_scores=confidence_scores,
            true_labels=true_labels
        )
        
        # Add metrics to metadata if calculated
        if metrics:
            if result.metadata is None:
                result.metadata = {}
            result.metadata['metrics'] = metrics
        
        return result

    def _calculate_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate evaluation metrics for binary vector predictions."""
        if self.classification_type == ClassificationType.MULTI_LABEL:
            return self._calculate_multi_label_metrics(predictions, true_labels)
        return self._calculate_single_label_metrics(predictions, true_labels)

    def _calculate_single_label_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate metrics for single-label classification using binary vectors."""
        if not predictions or not true_labels:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
        
        # Convert binary vectors to class indices for sklearn compatibility
        pred_classes = [pred.index(1) if 1 in pred else 0 for pred in predictions]
        true_classes = [true.index(1) if 1 in true else 0 for true in true_labels]
        
        # Calculate basic accuracy
        correct = sum(1 for pred, true in zip(pred_classes, true_classes) if pred == true)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        # For single-label classification with more than 2 classes, use macro averaging
        num_classes = len(self.classes_) if self.classes_ else max(max(pred_classes, default=0), max(true_classes, default=0)) + 1
        
        if num_classes <= 2:
            # Binary classification metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            try:
                precision = precision_score(true_classes, pred_classes, average='binary', zero_division=0)
                recall = recall_score(true_classes, pred_classes, average='binary', zero_division=0)
                f1 = f1_score(true_classes, pred_classes, average='binary', zero_division=0)
                
                # For AUC, we need probability scores, but we only have binary predictions
                # Use the prediction confidence as a proxy (1.0 for predicted class, 0.0 for others)
                try:
                    auc = roc_auc_score(true_classes, pred_classes)
                except ValueError:
                    # If all predictions are the same class, AUC is undefined
                    auc = 0.5
            except ImportError:
                # Fallback if sklearn is not available
                precision, recall, f1, auc = self._calculate_metrics_manual(pred_classes, true_classes, num_classes)
        else:
            # Multi-class classification metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            try:
                precision = precision_score(true_classes, pred_classes, average='macro', zero_division=0)
                recall = recall_score(true_classes, pred_classes, average='macro', zero_division=0)
                f1 = f1_score(true_classes, pred_classes, average='macro', zero_division=0)
                
                # For multi-class AUC, convert to one-hot and use ovr strategy
                try:
                    from sklearn.preprocessing import label_binarize
                    true_binary = label_binarize(true_classes, classes=list(range(num_classes)))
                    pred_binary = label_binarize(pred_classes, classes=list(range(num_classes)))
                    auc = roc_auc_score(true_binary, pred_binary, average='macro', multi_class='ovr')
                except (ValueError, ImportError):
                    auc = 0.5
            except ImportError:
                # Fallback if sklearn is not available
                precision, recall, f1, auc = self._calculate_metrics_manual(pred_classes, true_classes, num_classes)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def _calculate_metrics_manual(self, pred_classes: List[int], true_classes: List[int], num_classes: int) -> tuple:
        """Manual calculation of metrics when sklearn is not available."""
        # Calculate per-class metrics
        class_metrics = []
        
        for class_idx in range(num_classes):
            # True positives, false positives, false negatives for this class
            tp = sum(1 for pred, true in zip(pred_classes, true_classes) if pred == class_idx and true == class_idx)
            fp = sum(1 for pred, true in zip(pred_classes, true_classes) if pred == class_idx and true != class_idx)
            fn = sum(1 for pred, true in zip(pred_classes, true_classes) if pred != class_idx and true == class_idx)
            
            # Calculate precision, recall, f1 for this class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics.append((precision, recall, f1))
        
        # Macro average
        if class_metrics:
            avg_precision = sum(m[0] for m in class_metrics) / len(class_metrics)
            avg_recall = sum(m[1] for m in class_metrics) / len(class_metrics)
            avg_f1 = sum(m[2] for m in class_metrics) / len(class_metrics)
        else:
            avg_precision = avg_recall = avg_f1 = 0.0
        
        # Simple AUC approximation (not perfect but better than nothing)
        auc = 0.5  # Default for when we can't calculate properly
        
        return avg_precision, avg_recall, avg_f1, auc

    def _calculate_multi_label_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate metrics for multi-label classification using binary vectors."""
        if not predictions or not true_labels:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'hamming_loss': 1.0}
        
        # Sample-wise metrics
        sample_precisions = []
        sample_recalls = []
        sample_f1s = []
        exact_matches = 0
        hamming_distance = 0
        total_predictions = 0
        
        for pred, true in zip(predictions, true_labels):
            pred_set = set(i for i, val in enumerate(pred) if val == 1)
            true_set = set(i for i, val in enumerate(true) if val == 1)
            
            # Sample-wise precision, recall, F1
            if pred_set:
                precision = len(pred_set & true_set) / len(pred_set)
            else:
                precision = 1.0 if not true_set else 0.0
            
            if true_set:
                recall = len(pred_set & true_set) / len(true_set)
            else:
                recall = 1.0 if not pred_set else 0.0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            sample_precisions.append(precision)
            sample_recalls.append(recall)
            sample_f1s.append(f1)
            
            # Exact match (subset accuracy)
            if pred_set == true_set:
                exact_matches += 1
            
            # Hamming loss components
            for i in range(len(pred)):
                total_predictions += 1
                if pred[i] != true[i]:
                    hamming_distance += 1
        
        # Calculate averages
        avg_precision = sum(sample_precisions) / len(sample_precisions) if sample_precisions else 0.0
        avg_recall = sum(sample_recalls) / len(sample_recalls) if sample_recalls else 0.0
        avg_f1 = sum(sample_f1s) / len(sample_f1s) if sample_f1s else 0.0
        
        # Subset accuracy (exact match ratio)
        subset_accuracy = exact_matches / len(predictions) if predictions else 0.0
        
        # Hamming loss
        hamming_loss = hamming_distance / total_predictions if total_predictions > 0 else 0.0
        
        # Label-wise metrics (micro-averaged)
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            # Flatten for micro-averaging
            y_true_flat = [label for true in true_labels for label in true]
            y_pred_flat = [label for pred in predictions for label in pred]
            
            micro_precision = precision_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
            micro_recall = recall_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
            micro_f1 = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
            
        except ImportError:
            # Fallback when sklearn is not available
            micro_precision = avg_precision
            micro_recall = avg_recall
            micro_f1 = avg_f1
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'subset_accuracy': subset_accuracy,
            'hamming_loss': hamming_loss,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1
        }
