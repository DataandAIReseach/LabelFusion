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
        train_df,
        val_df = None
    ) -> Dict[str, Any]:
        """Train the RoBERTa classifier on the provided DataFrames or TrainingData.
        
        Args:
            train_df: Training DataFrame or TrainingData
            val_df: Optional validation DataFrame or TrainingData for evaluation during training
            
        Returns:
            Dictionary containing training metrics and information
        """
        from ..core.types import TrainingData
        
        # Handle different input types
        if isinstance(train_df, TrainingData):
            # For TrainingData, we don't need text_column and label_columns
            pass
        else:
            # For DataFrame, we need these parameters
            if not self.text_column:
                raise ValueError("text_column must be specified in constructor when using DataFrame")
            if not self.label_columns:
                raise ValueError("label_columns must be specified in constructor when using DataFrame")
            
            # Set up classes from label columns
            self.classes_ = self.label_columns
        
        # Validate input data
        self._validate_dataframe(train_df, self.text_column, self.label_columns, "training")
        if val_df is not None:
            self._validate_dataframe(val_df, self.text_column, self.label_columns, "validation")
        
        # Call the internal fit method
        self._fit_internal(train_df, val_df)
        
        # Auto-save model if path is provided
        if self.auto_save_path:
            self.save_model(self.auto_save_path)
        
        # Return training information
        training_samples = len(train_df.texts) if isinstance(train_df, TrainingData) else len(train_df)
        validation_samples = 0
        if val_df is not None:
            validation_samples = len(val_df.texts) if isinstance(val_df, TrainingData) else len(val_df)
        
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "classes": self.classes_,
            "training_samples": training_samples,
            "validation_samples": validation_samples,
            "device": str(self.device)
        }
    
    def predict(
        self,
        test_df: pd.DataFrame
    ) -> ClassificationResult:
        """Predict on test data using DataFrame.
        
        Note: Model must be trained first using the fit() method.
        
        Args:
            test_df: Test DataFrame for prediction
            
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
        
        # Make predictions on test data
        return self.predict_dataframes(test_df)
    
    def predict_dataframes(
        self,
        test_df: pd.DataFrame
    ) -> ClassificationResult:
        """Predict labels for test DataFrame.
        
        Args:
            test_df: Test DataFrame containing texts
            
        Returns:
            ClassificationResult with predictions and metrics
        """
        if not self.text_column:
            raise ValidationError("text_column must be specified in constructor")
        if not self.label_columns:
            raise ValidationError("label_columns must be specified in constructor")
            
        # Validate DataFrame
        self._validate_dataframe(test_df, self.text_column, self.label_columns, "test_df", require_labels=False)
        
        # Extract texts
        texts = test_df[self.text_column].tolist()
        
        # Extract true labels if available
        true_labels = None
        if all(col in test_df.columns for col in self.label_columns):
            true_labels = test_df[self.label_columns].values.tolist()
        
        # Make predictions
        return self.predict_texts(texts, true_labels)
    
    def _validate_dataframe(self, df, text_col: str, label_cols: List[str], df_name: str, require_labels: bool = True) -> None:
        """Validate that DataFrame or TrainingData has required columns/attributes."""
        from ..core.types import TrainingData
        
        if isinstance(df, TrainingData):
            # Handle TrainingData objects
            if not hasattr(df, 'texts') or not df.texts:
                raise ValidationError(f"{df_name} missing or empty texts attribute")
            
            if require_labels:
                if not hasattr(df, 'labels') or not df.labels:
                    raise ValidationError(f"{df_name} missing or empty labels attribute")
                
                # Check if we have the expected number of labels
                if len(df.texts) != len(df.labels):
                    raise ValidationError(f"{df_name} texts and labels must have the same length")
        
        elif hasattr(df, 'columns'):
            # Handle pandas DataFrame
            if text_col not in df.columns:
                raise ValidationError(f"{df_name} missing text column: {text_col}")
            
            if require_labels:
                missing_cols = [col for col in label_cols if col not in df.columns]
                if missing_cols:
                    raise ValidationError(f"{df_name} missing label columns: {missing_cols}")
        
        else:
            raise ValidationError(f"{df_name} must be either a pandas DataFrame or TrainingData object")
    
    def _determine_classification_type(self, df: pd.DataFrame, label_cols: List[str]) -> ClassificationType:
        """Determine if this is multi-class or multi-label classification."""
        if self.multi_label or len(label_cols) > 1:
            return ClassificationType.MULTI_LABEL
        else:
            return ClassificationType.MULTI_CLASS
    
    def _fit_internal(
        self, 
        train_df, 
        val_df = None
    ) -> None:
        """Internal method to train the RoBERTa classifier.
        
        Args:
            train_df: Training DataFrame or TrainingData
            val_df: Optional validation DataFrame or TrainingData
        """
        from ..core.types import TrainingData
        
        # Extract texts and labels based on input type
        if isinstance(train_df, TrainingData):
            # Handle TrainingData objects
            texts = train_df.texts
            labels = train_df.labels
            self.classification_type = train_df.classification_type
            
            # Convert string labels to binary if needed
            if labels and isinstance(labels[0], str):
                # Convert string labels to binary format
                unique_labels = list(set(labels))
                self.classes_ = unique_labels
                self.num_labels = len(unique_labels)
                
                # Create binary labels
                labels = []
                for label in train_df.labels:
                    binary_label = [0] * self.num_labels
                    if label in unique_labels:
                        binary_label[unique_labels.index(label)] = 1
                    labels.append(binary_label)
            else:
                # Labels are already in binary format
                if labels:
                    self.num_labels = len(labels[0])
                    if not self.label_columns:
                        self.classes_ = [f"class_{i}" for i in range(self.num_labels)]
                    else:
                        self.classes_ = self.label_columns
        
        else:
            # Handle pandas DataFrame
            # Determine classification type
            self.classification_type = self._determine_classification_type(train_df, self.label_columns)
            
            # Extract texts and labels from DataFrame
            texts = train_df[self.text_column].tolist()
            labels = train_df[self.label_columns].values.tolist()
            
            # Determine number of labels from the first label vector
            first_label = labels[0]
            self.num_labels = len(first_label)
            
            # Get class names from label_columns
            if self.label_columns:
                self.classes_ = self.label_columns
            else:
                if self.classification_type == ClassificationType.MULTI_CLASS:
                    self.classes_ = [f"class_{i}" for i in range(self.num_labels)]
                else:
                    self.classes_ = [f"label_{i}" for i in range(self.num_labels)]
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            cleaned = clean_text(text)
            normalized = normalize_text(cleaned)
            preprocessed = self.preprocessor.preprocess_text(normalized)
            processed_texts.append(preprocessed if preprocessed else text)  # Fallback to original if empty
        
        # Prepare labels - labels are already in binary format
        encoded_labels = labels
        
        # No label encoder needed since labels are already in binary format
        self.label_encoder = None
        
        # Initialize tokenizer and model
        try:
            # Validate that this is actually a RoBERTa model
            if not any(roberta_variant in self.model_name.lower() for roberta_variant in ['roberta', 'distilroberta']):
                raise ValueError(f"RoBERTaClassifier only supports RoBERTa models. Got: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Debug: Print label information
            print(f"Debug: num_labels = {self.num_labels}")
            print(f"Debug: classification_type = {self.classification_type}")
            print(f"Debug: classes = {self.classes_}")
            if encoded_labels:
                print(f"Debug: First few encoded labels: {encoded_labels[:3]}")
                print(f"Debug: Label shape example: {len(encoded_labels[0]) if encoded_labels[0] else 'None'}")
            
            # Use AutoModelForSequenceClassification for both multi-class and multi-label
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="multi_label_classification" if self.classification_type == ClassificationType.MULTI_LABEL else "single_label_classification"
            )
            
            self.model.to(self.device)
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to initialize RoBERTa model: {str(e)}", self.model_name)
        
        # Create dataset and dataloader
        dataset = TextDataset(processed_texts, encoded_labels, self.tokenizer, self.max_length, self.classification_type)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create validation dataloader if validation data is provided
        val_dataloader = None
        if val_df is not None:
            if isinstance(val_df, TrainingData):
                # Handle TrainingData objects
                val_texts = val_df.texts
                val_labels = val_df.labels
                
                # Convert string labels to binary if needed
                if val_labels and isinstance(val_labels[0], str):
                    val_binary_labels = []
                    for label in val_labels:
                        binary_label = [0] * self.num_labels
                        if label in self.classes_:
                            binary_label[self.classes_.index(label)] = 1
                        val_binary_labels.append(binary_label)
                    val_labels = val_binary_labels
            else:
                # Handle pandas DataFrame
                val_texts = val_df[self.text_column].tolist()
                val_labels = val_df[self.label_columns].values.tolist()
            
            # Preprocess validation texts
            processed_val_texts = []
            for text in val_texts:
                cleaned = clean_text(text)
                normalized = normalize_text(cleaned)
                preprocessed = self.preprocessor.preprocess_text(normalized)
                processed_val_texts.append(preprocessed if preprocessed else text)
            
            val_dataset = TextDataset(processed_val_texts, val_labels, self.tokenizer, self.max_length, self.classification_type)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        total_steps = len(dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.num_epochs):
            # Training phase
            total_loss = 0
            
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # Validation phase
            val_loss = None
            val_accuracy = None
            if val_dataloader is not None:
                val_loss, val_accuracy = self._evaluate_on_validation(val_dataloader)
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {avg_loss:.4f}")
        
        self.is_trained = True
    
    def _evaluate_on_validation(self, val_dataloader: DataLoader) -> tuple:
        """Evaluate the model on validation data.
        
        Args:
            val_dataloader: DataLoader for validation data
            
        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        self.model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                total_val_loss += loss.item()
                
                # Calculate accuracy
                if self.classification_type == ClassificationType.MULTI_CLASS:
                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)
                else:
                    # Multi-label: use threshold for binary predictions
                    threshold = 0.5
                    probabilities = torch.sigmoid(logits)
                    predictions = (probabilities > threshold).float()
                    
                    # Debug: Check tensor shapes
                    print(f"=== SHAPE DEBUG ===")
                    print(f"Predictions shape: {predictions.shape}")
                    print(f"Labels shape: {labels.shape}")
                    print(f"Logits shape: {logits.shape}")
                    print(f"Expected num_labels: {self.num_labels}")
                    print(f"First prediction: {predictions[0] if len(predictions) > 0 else 'None'}")
                    print(f"First label: {labels[0] if len(labels) > 0 else 'None'}")
                    
                    if predictions.shape != labels.shape:
                        print(f"Shape mismatch: predictions {predictions.shape}, labels {labels.shape}")
                        
                        # Handle shape mismatch by adjusting predictions to match labels
                        if predictions.shape[1] != labels.shape[1]:
                            # If predictions has fewer dimensions, pad with zeros
                            if predictions.shape[1] < labels.shape[1]:
                                padding = torch.zeros(predictions.shape[0], labels.shape[1] - predictions.shape[1], device=predictions.device)
                                predictions = torch.cat([predictions, padding], dim=1)
                                print(f"Padded predictions to shape: {predictions.shape}")
                            # If predictions has more dimensions, truncate
                            elif predictions.shape[1] > labels.shape[1]:
                                predictions = predictions[:, :labels.shape[1]]
                                print(f"Truncated predictions to shape: {predictions.shape}")
                    print(f"=== END DEBUG ===")
                    
                    # Exact match accuracy for multi-label
                    exact_matches = (predictions == labels).all(dim=1).sum().item()
                    correct_predictions += exact_matches
                    total_predictions += labels.size(0)
        
        self.model.train()  # Switch back to training mode
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_val_loss, val_accuracy
    
    def predict_texts(self, texts: List[str], true_labels: Optional[List[List[int]]] = None) -> ClassificationResult:
        """Predict labels for a list of texts (backward compatibility method).
        
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
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model and associated metadata.
        
        Args:
            save_path: Directory path where the model will be saved
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create save directory if it doesn't exist
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the model and tokenizer using transformers save_pretrained
        model_dir = save_dir / "model"
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "classes_": self.classes_,
            "text_column": self.text_column,
            "label_columns": self.label_columns,
            "classification_type": self.classification_type.value if self.classification_type else None,
            "multi_label": self.multi_label,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "device": str(self.device)
        }
        
        metadata_path = save_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """Load a previously saved model and metadata.
        
        Args:
            load_path: Directory path where the model was saved
        """
        load_dir = Path(load_path)
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {load_path}")
        
        # Load metadata
        metadata_path = load_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Restore instance attributes
        self.model_name = metadata["model_name"]
        self.num_labels = metadata["num_labels"]
        self.classes_ = metadata["classes_"]
        self.text_column = metadata["text_column"]
        self.label_columns = metadata["label_columns"]
        self.classification_type = ClassificationType(metadata["classification_type"]) if metadata["classification_type"] else None
        self.multi_label = metadata["multi_label"]
        self.max_length = metadata["max_length"]
        self.learning_rate = metadata["learning_rate"]
        self.batch_size = metadata["batch_size"]
        self.num_epochs = metadata["num_epochs"]
        
        # Load model and tokenizer
        model_dir = load_dir / "model"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.is_trained = True
            print(f"Model loaded from {load_path}")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to load model: {str(e)}", self.model_name)
    
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