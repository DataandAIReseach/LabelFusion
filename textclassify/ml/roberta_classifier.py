"""RoBERTa-based text classifier using transformers."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, List, Optional, Union
import warnings
import os
import json
from pathlib import Path

from ..core.types import ClassificationResult, ClassificationType, ModelType
from ..core.exceptions import ModelTrainingError, PredictionError, ValidationError
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
        if self.classification_type == ClassificationType.MULTI_LABEL:
            # Multi-label needs float for BCEWithLogitsLoss
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            # Multi-class: convert one-hot encoded vector to class index
            # e.g., [0, 1, 0] -> 1 (index of the 1)
            if isinstance(self.labels[idx], list):
                class_idx = self.labels[idx].index(1)  # Find index of the 1 in one-hot vector
            else:
                class_idx = int(np.argmax(self.labels[idx]))  # For numpy arrays
            label_tensor = torch.tensor(class_idx, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
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
        auto_save_path: Optional[str] = None
    ):
        """Initialize RoBERTa classifier.
        
        Args:
            config: Model configuration
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier
            enable_validation: Whether to evaluate on validation data during training
            auto_save_path: Optional path to automatically save model after training
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
        self.model.to(self.device)
        
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
        
        # Training loop
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_steps = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
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
            
            avg_train_loss = total_train_loss / train_steps
            print(f"  Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_loader is not None and self.enable_validation:
                self.model.eval()
                total_val_loss = 0
                val_steps = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
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
        
        # Return training information
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "classes": self.classes_,
            "training_samples": len(train_df),
            "validation_samples": len(val_df) if val_df is not None else 0,
            "device": str(self.device)
        }
    
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
        # Check if model is trained
        if not self.is_trained:
            raise ValidationError("Model must be trained first. Call fit() method before predict().")
        
        if not self.text_column:
            raise ValidationError("text_column must be specified in constructor")
        if not self.label_columns:
            raise ValidationError("label_columns must be specified in constructor")
        
        # Extract texts
        texts = test_df[self.text_column].tolist()
        
        # Extract true labels if available
        true_labels = None
        if all(col in test_df.columns for col in self.label_columns):
            true_labels = test_df[self.label_columns].values.tolist()
        
        # Make predictions using the internal text method
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
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                if self.classification_type == ClassificationType.MULTI_CLASS:
                    predictions = torch.argmax(logits, dim=-1)
                    
                    # Convert predictions to class names
                    for pred_idx in predictions.cpu().numpy():
                        if pred_idx < len(self.classes_):
                            all_predictions.append(self.classes_[pred_idx])
                        else:
                            # Fallback: use first class if prediction is out of bounds
                            all_predictions.append(self.classes_[0])
                else:
                    # Multi-label classification
                    probabilities = torch.sigmoid(logits)
                    threshold = self.config.parameters.get('threshold', 0.5)
                    predictions = (probabilities > threshold).cpu().numpy()
                    
                    # Convert predictions to class names
                    for pred_array in predictions:
                        active_classes = []
                        for i, is_active in enumerate(pred_array):
                            if is_active and i < len(self.classes_):
                                active_classes.append(self.classes_[i])
                        all_predictions.append(active_classes)
        
        # Calculate metrics if true labels are provided
        return self._create_result(
            predictions=all_predictions, 
            true_labels=true_labels if true_labels is not None else None
        )
    
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
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
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
                    threshold = self.config.parameters.get('threshold', 0.5)
                    predictions = (probabilities > threshold).cpu().numpy()
                    batch_probabilities = probabilities.cpu().numpy()
                    
                    # Convert predictions to class names using self.classes_
                    for i, pred_array in enumerate(predictions):
                        active_labels = [self.classes_[j] for j, is_active in enumerate(pred_array) if is_active]
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

