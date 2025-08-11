"""RoBERTa-based text classifier using transformers."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, List, Optional, Union
import warnings

from ..core.types import ClassificationResult, ClassificationType, TrainingData
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
            # Multi-class needs long for CrossEntropyLoss
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }


class RoBERTaClassifier(BaseMLClassifier):
    """RoBERTa-based text classifier."""
    
    def __init__(self, config):
        """Initialize RoBERTa classifier.
        
        Args:
            config: Model configuration
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and scikit-learn are required for RoBERTa classifier. "
                "Install with: pip install transformers torch scikit-learn"
            )
        
        super().__init__(config)
        
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
    
    def fit(self, training_data: TrainingData) -> None:
        """Train the RoBERTa classifier.
        
        Args:
            training_data: Training data containing texts and labels
        """
        self._validate_training_data(training_data)
        
        self.classification_type = training_data.classification_type
        
        # Preprocess texts
        processed_texts = []
        for text in training_data.texts:
            cleaned = clean_text(text)
            normalized = normalize_text(cleaned)
            preprocessed = self.preprocessor.preprocess_text(normalized)
            processed_texts.append(preprocessed if preprocessed else text)  # Fallback to original if empty
        
        # Prepare labels
        if self.classification_type == ClassificationType.MULTI_CLASS:
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(training_data.labels)
            self.classes_ = self.label_encoder.classes_.tolist()
            self.num_labels = len(self.classes_)
        else:
            # Multi-label classification
            self.label_encoder = MultiLabelBinarizer()
            encoded_labels = self.label_encoder.fit_transform(training_data.labels)
            self.classes_ = self.label_encoder.classes_.tolist()
            self.num_labels = len(self.classes_)
        
        # Initialize tokenizer and model
        try:
            # Validate that this is actually a RoBERTa model
            if not any(roberta_variant in self.model_name.lower() for roberta_variant in ['roberta', 'distilroberta']):
                raise ValueError(f"RoBERTaClassifier only supports RoBERTa models. Got: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
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
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Loss: {avg_loss:.4f}")
        
        self.is_trained = True
    
    def predict(self, texts: List[str]) -> ClassificationResult:
        """Predict labels for texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            ClassificationResult with predictions
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
                    batch_predictions = self.label_encoder.inverse_transform(predictions.cpu().numpy())
                    all_predictions.extend(batch_predictions)
                else:
                    # Multi-label classification
                    probabilities = torch.sigmoid(logits)
                    threshold = self.config.parameters.get('threshold', 0.5)
                    predictions = (probabilities > threshold).cpu().numpy()
                    batch_predictions = self.label_encoder.inverse_transform(predictions)
                    all_predictions.extend(batch_predictions.tolist())
        
        return self._create_result(predictions=all_predictions)
    
    def predict_proba(self, texts: List[str]) -> ClassificationResult:
        """Predict class probabilities for texts.
        
        Args:
            texts: List of texts to classify
            
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
                    
                    batch_predictions = self.label_encoder.inverse_transform(predictions.cpu().numpy())
                    batch_probabilities = probabilities.cpu().numpy()
                    
                    for i, pred in enumerate(batch_predictions):
                        all_predictions.append(pred)
                        
                        # Create probability dictionary
                        prob_dict = {
                            class_name: float(batch_probabilities[i][j])
                            for j, class_name in enumerate(self.classes_)
                        }
                        all_probabilities.append(prob_dict)
                        all_confidence_scores.append(float(batch_probabilities[i][predictions[i]]))
                
                else:
                    # Multi-label classification
                    probabilities = torch.sigmoid(logits)
                    threshold = self.config.parameters.get('threshold', 0.5)
                    predictions = (probabilities > threshold).cpu().numpy()
                    
                    batch_predictions = self.label_encoder.inverse_transform(predictions)
                    batch_probabilities = probabilities.cpu().numpy()
                    
                    for i, pred in enumerate(batch_predictions):
                        all_predictions.append(pred.tolist())
                        
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
            confidence_scores=all_confidence_scores
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

