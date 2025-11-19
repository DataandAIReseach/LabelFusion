"""RoBERTa-based text classifier using transformers."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import faulthandler
import gc
from typing import Any, Dict, List, Optional, Union
import warnings
import os
import json
from pathlib import Path

from ..core.types import ClassificationResult, ClassificationType, ModelType
from ..core.exceptions import ModelTrainingError, PredictionError, ValidationError, CudaOutOfMemoryError
from ..utils.results_manager import ResultsManager, ModelResultsManager
from .base import BaseMLClassifier
from .preprocessing import TextPreprocessor, clean_text, normalize_text

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

try:
    # Use Auto* classes which are the recommended approach for modern transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup
    )
    from torch.optim import AdamW
    from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    TRANSFORMERS_AVAILABLE = False



class TextDataset(Dataset):
    """Dataset class for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512, classification_type=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.classification_type = classification_type
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Use appropriate data type based on classification type
        # Normalize label formats robustly to avoid crashes when label arrays contain no explicit 1
        import numpy as _np

        raw_label = self.labels[idx]

        # Convert various possible label representations to a numpy array
        try:
            if isinstance(raw_label, list):
                label_arr = _np.array(raw_label)
            elif hasattr(raw_label, 'dtype') or hasattr(raw_label, 'shape'):
                # numpy array or pandas-backed type
                label_arr = _np.array(raw_label)
            else:
                # scalar (class index) -> wrap into array
                label_arr = _np.array([raw_label])
        except Exception:
            # Fallback: try to coerce to array
            label_arr = _np.array(raw_label)

        if self.classification_type == ClassificationType.MULTI_LABEL:
            # Multi-label expects a binary vector
            label_tensor = torch.tensor(label_arr.astype(float), dtype=torch.float)
        else:
            # MULTI_CLASS: prefer explicit one-hot (index of 1). If no 1 present, fallback to argmax.
            if label_arr.size > 1:
                ones = _np.where(label_arr == 1)[0]
                if ones.size > 0:
                    class_idx = int(ones[0])
                else:
                    # no explicit one-hot marker found -> fallback to argmax
                    class_idx = int(_np.argmax(label_arr))
            else:
                # already a scalar class index
                class_idx = int(label_arr.item())

            label_tensor = torch.tensor(class_idx, dtype=torch.long)

        # Remove leading batch dim from tokenizer outputs (tokenizers return tensors with batch dim)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_tensor
        }


class RoBERTaClassifier(BaseMLClassifier):
    """RoBERTa-based text classifier."""
    
    def __init__(
        self,
        config,
        text_column: str = 'text',
        label_columns: Optional[List[str]] = None,
        multi_label: bool = False,
        enable_validation: bool = True,
        auto_save_path: Optional[str] = None,
        auto_save_results: bool = True,
        output_dir: str = "outputs",
        experiment_name: Optional[str] = None
    ):
        """Initialize RoBERTa classifier.
        
        Args:
            config: Model configuration
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier
            enable_validation: Whether to evaluate on validation data during training
            auto_save_path: Optional path to automatically save model after training
            auto_save_results: Whether to automatically save training/prediction results
            output_dir: Base directory for saving results
            experiment_name: Name for the experiment (for results organization)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and scikit-learn are required for RoBERTa classifier. "
                "Install with: pip install transformers torch scikit-learn"
            )
        
        super().__init__(config)
        
        # DataFrame interface parameters
        self.text_column = text_column
        self.label_columns = label_columns or []
        self.multi_label = multi_label
        self.enable_validation = enable_validation
        self.auto_save_path = auto_save_path
        
        # Results management
        self.auto_save_results = auto_save_results
        self.results_manager = None
        self.model_results_manager = None
        
        if self.auto_save_results:
            exp_name = experiment_name or f"roberta_{self.config.model_name}"
            self.results_manager = ResultsManager(
                base_output_dir=output_dir,
                experiment_name=exp_name
            )
            self.model_results_manager = ModelResultsManager(
                self.results_manager,
                f"roberta_{self.config.model_name}"
            )
        
        # Set up classes
        self.classes_ = label_columns if label_columns else []
        
        # Model parameters
        self.model_name = self.config.parameters.get('model_name', 'roberta-base')
        self.max_length = self.config.parameters.get('max_length', 512)
        self.batch_size = self.config.parameters.get('batch_size', 16)
        self.learning_rate = self.config.parameters.get('learning_rate', 2e-5)
        self.num_epochs = self.config.parameters.get('num_epochs', 3)
        self.warmup_steps = self.config.parameters.get('warmup_steps', 0)
        self.weight_decay = self.config.parameters.get('weight_decay', 0.01)
        
        # Preprocessing
        preprocessing_config = self.config.parameters.get('preprocessing', {})
        self.preprocessor = TextPreprocessor(**preprocessing_config)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.num_labels = None
    
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Train the RoBERTa classifier on the provided DataFrames.
        
        Args:
            train_df: Training DataFrame with text and label columns
            val_df: Optional validation DataFrame for evaluation during training
            
        Returns:
            Dictionary containing training metrics and information
        """
        if not self.text_column:
            raise ValueError("text_column must be specified in constructor")
        if not self.label_columns:
            raise ValueError("label_columns must be specified in constructor")
        
        # Set up classes from label columns
        self.classes_ = self.label_columns
        
        # Extract and preprocess texts from DataFrames
        train_texts = train_df[self.text_column].tolist()
        train_texts = [self.preprocessor.preprocess_text(text) for text in train_texts]
        train_labels = train_df[self.label_columns].values.tolist()
        
        val_texts = None
        val_labels = None
        if val_df is not None:
            val_texts = val_df[self.text_column].tolist()
            val_texts = [self.preprocessor.preprocess_text(text) for text in val_texts]
            val_labels = val_df[self.label_columns].values.tolist()
        
        # Set up tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Determine number of labels and classification type
        self.num_labels = len(self.label_columns)
        
        # Set classification type based on multi_label flag
        if self.multi_label:
            self.classification_type = ClassificationType.MULTI_LABEL
        else:
            self.classification_type = ClassificationType.MULTI_CLASS
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification" if self.multi_label else "single_label_classification"
        )
        # Move model to device with explicit OOM handling
        try:
            self.model.to(self.device)
        except Exception as e:
            if 'out of memory' in str(e).lower():
                # Raise a structured CUDA OOM exception with parsed details
                self._handle_cuda_oom(e, context="moving model to device")
            raise
        
        # Create datasets
        train_dataset = TextDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            classification_type=self.classification_type
        )
        
        val_dataset = None
        if val_texts is not None and val_labels is not None:
            val_dataset = TextDataset(
                texts=val_texts,
                labels=val_labels,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                classification_type=self.classification_type
            )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Set up optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Ensure faulthandler is enabled to get Python-level tracebacks on crashes
        try:
            faulthandler.enable()
        except Exception:
            # faulthandler may already be enabled or not available in some envs
            pass

        # Training loop
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Training phase
            self.model.train()
            total_train_loss = 0
            train_steps = 0

            # Basic sanity prints
            try:
                print(f"Train loader batches: {len(train_loader)} | batch_size: {self.batch_size} | device: {self.device}")
            except Exception:
                print("Train loader length not available")

            for batch_idx, batch in enumerate(train_loader):
                # Periodic logging to trace progress and detect silent exits
                if batch_idx % 10 == 0:
                    print(f"  [epoch {epoch+1}] processing batch {batch_idx}")

                try:
                    input_ids = self._to_device(batch['input_ids'], name='input_ids')
                    attention_mask = self._to_device(batch['attention_mask'], name='attention_mask')
                    labels = self._to_device(batch['labels'], name='labels')

                    optimizer.zero_grad()

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_train_loss += loss.item()
                    train_steps += 1

                except Exception as e:
                    # Catch and log any exception during batch processing so we don't exit silently
                    import traceback as _tb
                    print(f"Exception while processing batch {batch_idx}: {e}")
                    _tb.print_exc()
                    # Try to free GPU memory and continue or re-raise depending on severity
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()
                    # Re-raise to stop training after logging — caller can inspect logs
                    raise
            
            avg_train_loss = total_train_loss / train_steps
            print(f"  Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_loader is not None and self.enable_validation:
                self.model.eval()
                total_val_loss = 0
                val_steps = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = self._to_device(batch['input_ids'], name='input_ids')
                        attention_mask = self._to_device(batch['attention_mask'], name='attention_mask')
                        labels = self._to_device(batch['labels'], name='labels')
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        total_val_loss += loss.item()
                        val_steps += 1
                
                avg_val_loss = total_val_loss / val_steps
                print(f"  Average validation loss: {avg_val_loss:.4f}")
        
        self.is_trained = True
        print("✅ Model training completed!")
        
        # Auto-save model if path is provided
        if self.auto_save_path:
            self.save_model(self.auto_save_path)
        
        # Generate and save validation predictions if validation data is provided
        val_predictions_saved = None
        if val_df is not None and self.enable_validation:
            print("📝 Generating validation predictions...")
            try:
                val_result = self._predict_on_dataset(val_df, dataset_type="validation")
                val_predictions_saved = {
                    'accuracy': val_result.metadata.get('metrics', {}).get('accuracy', 0.0) if val_result.metadata else 0.0,
                    'num_samples': len(val_df)
                }
                print(f"✅ Validation predictions saved with accuracy: {val_predictions_saved['accuracy']:.4f}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save validation predictions: {e}")
                val_predictions_saved = None
        
        # Prepare training result
        training_result = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "classes": self.classes_,
            "training_samples": len(train_df),
            "validation_samples": len(val_df) if val_df is not None else 0,
            "validation_predictions": val_predictions_saved,
            "device": str(self.device)
        }
        
        # Save training results using ResultsManager
        if self.results_manager:
            try:
                # Save model configuration
                model_config_dict = {
                    'model_name': self.model_name,
                    'max_length': self.max_length,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'num_epochs': self.num_epochs,
                    'multi_label': self.multi_label,
                    'num_labels': self.num_labels,
                    'classes': self.classes_,
                    'text_column': self.text_column,
                    'label_columns': self.label_columns
                }
                
                self.results_manager.save_model_config(
                    model_config_dict, 
                    "roberta_classifier"
                )
                
                # Save training summary
                self.results_manager.save_experiment_summary(training_result)
                
                # Get experiment info
                exp_info = self.results_manager.get_experiment_info()
                training_result['experiment_info'] = exp_info
                training_result['output_directory'] = exp_info['experiment_dir']
                
                print(f"📁 Training results saved to: {exp_info['experiment_dir']}")
                
            except Exception as e:
                print(f"Warning: Could not save training results: {e}")
        
        return training_result
    
    def _predict_on_dataset(
        self, 
        data_df: pd.DataFrame, 
        dataset_type: str = "test"
    ) -> ClassificationResult:
        """Predict on a dataset with specified dataset type for proper file naming.
        
        Args:
            data_df: DataFrame for prediction with text and optionally label columns
            dataset_type: Type of dataset ("test", "validation", etc.) for file naming
            
        Returns:
            ClassificationResult with predictions and metrics
        """
        # Check if model is trained
        if not self.is_trained:
            raise ValidationError("Model must be trained first. Call fit() method before predict().")
        
        if not self.text_column:
            raise ValidationError("text_column must be specified in constructor")
        if not self.label_columns:
            raise ValidationError("label_columns must be specified in constructor")
        
        # Extract texts
        texts = data_df[self.text_column].tolist()
        
        # Extract true labels if available
        true_labels = None
        if all(col in data_df.columns for col in self.label_columns):
            true_labels = data_df[self.label_columns].values.tolist()
        
        # Store DataFrame reference for results saving with specified dataset type
        if self.results_manager:
            self._current_test_df = data_df
            self._current_dataset_type = dataset_type
        
        # Make predictions using the internal text method
        result = self._predict_texts_internal(texts, true_labels)
        
        # Clean up temporary references
        if hasattr(self, '_current_test_df'):
            delattr(self, '_current_test_df')
        if hasattr(self, '_current_dataset_type'):
            delattr(self, '_current_dataset_type')
            
        return result

    def predict(
        self,
        test_df: pd.DataFrame
    ) -> ClassificationResult:
        """Predict on test data using DataFrame.
        
        Note: Model must be trained first using the fit() method.
        
        Args:
            test_df: Test DataFrame for prediction with text and optionally label columns
            
        Returns:
            ClassificationResult with predictions and metrics
        """
        return self._predict_on_dataset(test_df, dataset_type="test")

    def predict_without_saving(
        self,
        data_df: pd.DataFrame
    ) -> ClassificationResult:
        """Predict on data without saving results (for internal use by ensembles).
        
        Args:
            data_df: DataFrame for prediction with text and optionally label columns
            
        Returns:
            ClassificationResult with predictions and metrics
        """
        # Check if model is trained
        if not self.is_trained:
            raise ValidationError("Model must be trained first. Call fit() method before predict().")
        
        if not self.text_column:
            raise ValidationError("text_column must be specified in constructor")
        if not self.label_columns:
            raise ValidationError("label_columns must be specified in constructor")
        
        # Extract texts
        texts = data_df[self.text_column].tolist()
        
        # Extract true labels if available
        true_labels = None
        if all(col in data_df.columns for col in self.label_columns):
            true_labels = data_df[self.label_columns].values.tolist()
        
        # Make predictions using the internal text method (no saving)
        return self._predict_texts_internal(texts, true_labels)

    def predict_texts(self, texts: List[str], true_labels: Optional[List[List[int]]] = None) -> ClassificationResult:
        """Predict labels for a list of texts (compatibility method for FusionEnsemble).
        
        This method is provided for compatibility with FusionEnsemble which calls 
        ML models with text lists. For regular usage, use predict(test_df) instead.
        
        Args:
            texts: List of texts to classify
            true_labels: Optional true labels in binary format for evaluation metrics
            
        Returns:
            ClassificationResult with predictions and optional metrics
        """
        return self._predict_texts_internal(texts, true_labels)
    
    def _predict_texts_internal(self, texts: List[str], true_labels: Optional[List[List[int]]] = None) -> ClassificationResult:
        """Internal method to predict labels for a list of texts.
        
        Args:
            texts: List of texts to classify
            true_labels: Optional true labels in binary format for evaluation metrics
            
        Returns:
            ClassificationResult with predictions and optional metrics
        """
        self.validate_input(texts)
        
        if not self.is_trained:
            raise PredictionError("Model must be trained before prediction", self.model_name)
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            cleaned = clean_text(text)
            normalized = normalize_text(cleaned)
            preprocessed = self.preprocessor.preprocess_text(normalized)
            processed_texts.append(preprocessed if preprocessed else text)
        
        # Create dataset and dataloader
        if self.classification_type == ClassificationType.MULTI_LABEL:
            dummy_labels = [[0] * self.num_labels] * len(processed_texts)  # Multi-label dummy labels
        else:
            dummy_labels = [0] * len(processed_texts)  # Multi-class dummy labels
        dataset = TextDataset(processed_texts, dummy_labels, self.tokenizer, self.max_length, self.classification_type)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Prediction
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_embeddings = []  # Store embeddings for fusion ensemble
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = self._to_device(batch['input_ids'], name='input_ids')
                attention_mask = self._to_device(batch['attention_mask'], name='attention_mask')
                
                # Get both logits and hidden states (embeddings)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                logits = outputs.logits
                hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
                
                # Extract [CLS] token embeddings (first token)
                cls_embeddings = hidden_states[:, 0, :]  # Shape: [batch_size, 768]
                all_embeddings.extend(cls_embeddings.cpu().numpy())
                
                if self.classification_type == ClassificationType.MULTI_CLASS:
                    # Get probabilities using softmax
                    probs = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)
                    
                    # Convert predictions to class names and probabilities to dicts
                    for pred_idx, prob_vector in zip(predictions.cpu().numpy(), probs.cpu().numpy()):
                        if pred_idx < len(self.classes_):
                            all_predictions.append(self.classes_[pred_idx])
                        else:
                            # Fallback: use first class if prediction is out of bounds
                            all_predictions.append(self.classes_[0])
                        
                        # Create probability dictionary for all classes
                        prob_dict = {self.classes_[i]: float(prob_vector[i]) for i in range(len(self.classes_))}
                        all_probabilities.append(prob_dict)
                else:
                    # Multi-label classification
                    probabilities = torch.sigmoid(logits)
                    threshold = self.config.parameters.get('threshold', 0.3)
                    predictions = (probabilities > threshold).cpu().numpy()
                    
                    # Convert predictions to class names with dynamic fallback
                    for pred_array, prob_vector in zip(predictions, probabilities.cpu().numpy()):
                        active_classes = []
                        for i, is_active in enumerate(pred_array):
                            if is_active and i < len(self.classes_):
                                active_classes.append(self.classes_[i])
                        
                        # Dynamic fallback: if no predictions above threshold, take top label
                        if len(active_classes) == 0:
                            top_idx = prob_vector.argmax()
                            if top_idx < len(self.classes_):
                                active_classes = [self.classes_[top_idx]]
                        
                        all_predictions.append(active_classes)
                        
                        # Create probability dictionary for all classes
                        prob_dict = {self.classes_[i]: float(prob_vector[i]) for i in range(len(self.classes_))}
                        all_probabilities.append(prob_dict)
        
        # Calculate metrics if true labels are provided
        result = self._create_result(
            predictions=all_predictions,
            probabilities=all_probabilities if all_probabilities else None,
            true_labels=true_labels if true_labels is not None else None,
            embeddings=all_embeddings if all_embeddings else None  # Add embeddings to result
        )
        
        # Save prediction results using ResultsManager (if it's the main predict call)
        if self.results_manager and hasattr(self, '_current_test_df'):
            try:
                # Use dataset type if available, otherwise default to "test"
                dataset_type = getattr(self, '_current_dataset_type', 'test')
                
                saved_files = self.results_manager.save_predictions(
                    result, dataset_type, self._current_test_df
                )
                
                # Save metrics if available
                if hasattr(result, 'metadata') and result.metadata and 'metrics' in result.metadata:
                    metrics_file = self.results_manager.save_metrics(
                        result.metadata['metrics'], dataset_type, "roberta_classifier"
                    )
                    saved_files["metrics"] = metrics_file
                
                print(f"📁 Prediction results saved: {saved_files}")
                
                # Add file paths to result metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata['saved_files'] = saved_files
                
                # Clean up temporary references
                if hasattr(self, '_current_test_df'):
                    delattr(self, '_current_test_df')
                if hasattr(self, '_current_dataset_type'):
                    delattr(self, '_current_dataset_type')
                
            except Exception as e:
                print(f"Warning: Could not save prediction results: {e}")
        
        return result
    
    def _create_result(
        self,
        predictions: List[Union[str, List[str]]],
        probabilities: Optional[List[Dict[str, float]]] = None,
        confidence_scores: Optional[List[float]] = None,
        processing_time: Optional[float] = None,
        true_labels: Optional[List[List[int]]] = None,
        **metadata
    ) -> ClassificationResult:
        """Create a ClassificationResult with metrics calculation if true labels provided.
        
        Args:
            predictions: Predicted labels
            probabilities: Class probabilities (optional)
            confidence_scores: Confidence scores (optional)
            processing_time: Time taken for processing (optional)
            true_labels: True labels in binary format for evaluation metrics (optional)
            **metadata: Additional metadata
            
        Returns:
            ClassificationResult with populated metadata and optional metrics
        """
        # Calculate metrics if true labels are provided
        if true_labels is not None:
            # Convert predictions back to binary format for metric calculation
            predicted_labels = []
            for pred in predictions:
                if isinstance(pred, str):
                    # Multi-class: convert class name back to one-hot
                    pred_vector = [0] * len(self.classes_)
                    if pred in self.classes_:
                        pred_idx = self.classes_.index(pred)
                        pred_vector[pred_idx] = 1
                    predicted_labels.append(pred_vector)
                else:
                    # Multi-label: convert class names back to binary
                    pred_vector = [0] * len(self.classes_)
                    for class_name in pred:
                        if class_name in self.classes_:
                            pred_idx = self.classes_.index(class_name)
                            pred_vector[pred_idx] = 1
                    predicted_labels.append(pred_vector)
            
            # Calculate metrics
            import numpy as np
            from sklearn.metrics import accuracy_score, classification_report, hamming_loss, precision_recall_fscore_support
            
            true_labels_array = np.array(true_labels)
            predicted_labels_array = np.array(predicted_labels)
            
            metrics = {}
            
            if self.classification_type == ClassificationType.MULTI_CLASS:
                # Multi-class metrics
                true_indices = np.argmax(true_labels_array, axis=1)
                pred_indices = np.argmax(predicted_labels_array, axis=1)
                
                accuracy = accuracy_score(true_indices, pred_indices)
                precision, recall, f1, support = precision_recall_fscore_support(
                    true_indices, pred_indices, average='weighted', zero_division=0
                )
                
                metrics.update({
                    'accuracy': float(accuracy),
                    'precision_weighted': float(precision),
                    'recall_weighted': float(recall),
                    'f1_weighted': float(f1),
                    'classification_report': classification_report(
                        true_indices, pred_indices, 
                        labels=list(range(len(self.classes_))),
                        target_names=self.classes_, 
                        output_dict=True,
                        zero_division=0
                    )
                })
            else:
                # Multi-label metrics
                exact_match_accuracy = accuracy_score(true_labels_array, predicted_labels_array)
                hamming = hamming_loss(true_labels_array, predicted_labels_array)
                
                # Per-label metrics
                precision, recall, f1, support = precision_recall_fscore_support(
                    true_labels_array, predicted_labels_array, average='weighted', zero_division=0
                )
                
                metrics.update({
                    'exact_match_accuracy': float(exact_match_accuracy),
                    'hamming_loss': float(hamming),
                    'precision_weighted': float(precision),
                    'recall_weighted': float(recall),
                    'f1_weighted': float(f1),
                    'classification_report': classification_report(
                        true_labels_array, predicted_labels_array, 
                        target_names=self.classes_, 
                        output_dict=True,
                        zero_division=0
                    )
                })
            
            # Add metrics to metadata
            metadata['metrics'] = metrics
        
        # Call parent _create_result and then add metadata
        result = super()._create_result(
            predictions=predictions,
            probabilities=probabilities,
            confidence_scores=confidence_scores
        )
        
        # Add metadata to the result
        if result.metadata is None:
            result.metadata = {}
        result.metadata.update(metadata)
        
        return result
    
    def predict_proba(
        self, 
        test_df: pd.DataFrame
    ) -> ClassificationResult:
        """Predict class probabilities for DataFrame.
        
        Args:
            test_df: Test DataFrame for prediction with text and optionally label columns
            
        Returns:
            ClassificationResult with predictions and probabilities
        """
        if not self.text_column:
            raise ValidationError("text_column must be specified in constructor")
        if not self.label_columns:
            raise ValidationError("label_columns must be specified in constructor")
        
        # Extract texts and labels
        texts = test_df[self.text_column].tolist()
        true_labels = None
        if all(col in test_df.columns for col in self.label_columns):
            true_labels = test_df[self.label_columns].values.tolist()
        
        # Call the internal prediction method
        return self._predict_proba_texts(texts, true_labels)
    
    def _predict_proba_texts(self, texts: List[str], true_labels: Optional[List[List[int]]] = None) -> ClassificationResult:
        """Predict class probabilities for texts.
        
        Args:
            texts: List of texts to classify
            true_labels: Optional true labels in binary format for evaluation metrics
            
        Returns:
            ClassificationResult with predictions and probabilities
        """
        self.validate_input(texts)
        
        if not self.is_trained:
            raise PredictionError("Model must be trained before prediction", self.model_name)
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            cleaned = clean_text(text)
            normalized = normalize_text(cleaned)
            preprocessed = self.preprocessor.preprocess_text(normalized)
            processed_texts.append(preprocessed if preprocessed else text)
        
        # Create dataset and dataloader
        if self.classification_type == ClassificationType.MULTI_LABEL:
            dummy_labels = [[0] * self.num_labels] * len(processed_texts)  # Multi-label dummy labels
        else:
            dummy_labels = [0] * len(processed_texts)  # Multi-class dummy labels
        dataset = TextDataset(processed_texts, dummy_labels, self.tokenizer, self.max_length, self.classification_type)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Prediction with probabilities
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_confidence_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = self._to_device(batch['input_ids'], name='input_ids')
                attention_mask = self._to_device(batch['attention_mask'], name='attention_mask')
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                if self.classification_type == ClassificationType.MULTI_CLASS:
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1)
                    batch_probabilities = probabilities.cpu().numpy()
                    
                    # Convert predictions to class names using self.classes_
                    for i, pred_idx in enumerate(predictions.cpu().numpy()):
                        if pred_idx < len(self.classes_):
                            pred_name = self.classes_[pred_idx]
                        else:
                            pred_name = f"class_{pred_idx}"  # Fallback
                        
                        all_predictions.append(pred_name)
                        
                        # Create probability dictionary
                        prob_dict = {
                            class_name: float(batch_probabilities[i][j])
                            for j, class_name in enumerate(self.classes_)
                        }
                        all_probabilities.append(prob_dict)
                        all_confidence_scores.append(float(batch_probabilities[i][pred_idx]))
                
                else:
                    # Multi-label classification
                    probabilities = torch.sigmoid(logits)
                    threshold = self.config.parameters.get('threshold', 0.3)
                    predictions = (probabilities > threshold).cpu().numpy()
                    batch_probabilities = probabilities.cpu().numpy()
                    
                    # Convert predictions to class names using self.classes_ with dynamic fallback
                    for i, pred_array in enumerate(predictions):
                        active_labels = [self.classes_[j] for j, is_active in enumerate(pred_array) if is_active]
                        
                        # Dynamic fallback: if no predictions above threshold, take top label
                        if len(active_labels) == 0:
                            top_idx = batch_probabilities[i].argmax()
                            if top_idx < len(self.classes_):
                                active_labels = [self.classes_[top_idx]]
                        
                        all_predictions.append(active_labels)
                        
                        # Create probability dictionary
                        prob_dict = {
                            class_name: float(batch_probabilities[i][j])
                            for j, class_name in enumerate(self.classes_)
                        }
                        all_probabilities.append(prob_dict)
                        
                        # Confidence is max probability for multi-label
                        all_confidence_scores.append(float(np.max(batch_probabilities[i])))
        
        return self._create_result(
            predictions=all_predictions,
            probabilities=all_probabilities,
            confidence_scores=all_confidence_scores,
            true_labels=true_labels if true_labels is not None else None
        )

    # --- Helper methods for robust CUDA OOM handling ---
    def _handle_cuda_oom(self, exc: Exception, context: str = ""):
        """Parse PyTorch CUDA OOM message and raise a structured CudaOutOfMemoryError.

        The parser attempts to extract commonly included details from the
        PyTorch CUDA OOM message (attempted allocation, total/free memory,
        process id). These are best-effort and may be None if parsing fails.
        """
        import re

        msg = str(exc)
        attempted = None
        total = None
        free = None
        pid = None

        m = re.search(r"Tried to allocate ([0-9\.]+) MiB", msg)
        if m:
            attempted = f"{m.group(1)} MiB"

        m = re.search(r"has a total capacity of ([0-9\.]+) GiB", msg)
        if m:
            total = f"{m.group(1)} GiB"

        m = re.search(r"of which ([0-9\.]+) MiB is free", msg)
        if m:
            free = f"{m.group(1)} MiB"

        m = re.search(r"Process (\d+)", msg)
        if m:
            try:
                pid = int(m.group(1))
            except Exception:
                pid = None

        suggestion = (
            "Try freeing GPU memory, set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, "
            "reduce batch size, run on CPU, or reboot the node to clear defunct allocations."
        )

        raise CudaOutOfMemoryError(
            message=f"CUDA out of memory while {context}: {msg}",
            attempted_allocation=attempted,
            total_memory=total,
            free_memory=free,
            process_id=pid,
            suggestion=suggestion,
            model_name=getattr(self, 'model_name', None)
        ) from exc

    def _to_device(self, tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """Move a tensor to the configured device and convert CUDA OOM into a structured exception."""
        try:
            return tensor.to(self.device)
        except Exception as e:
            if 'out of memory' in str(e).lower():
                self._handle_cuda_oom(e, context=f"moving {name} to device")
            raise
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get RoBERTa model information."""
        info = super().model_info
        info.update({
            "provider": "huggingface",
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "device": str(self.device),
            "num_labels": self.num_labels
        })
        return info

