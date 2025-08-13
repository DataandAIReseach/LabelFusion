"""Fusion ensemble combining ML and LLM classifiers with trainable MLP."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

from ..core.types import ClassificationResult, ClassificationType, TrainingData, EnsembleConfig
from ..core.exceptions import EnsembleError, ModelTrainingError
from .base import BaseEnsemble


class FusionMLP(nn.Module):
    """Trainable MLP for fusing ML and LLM predictions."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [64, 32]):
        """Initialize Fusion MLP.
        
        Args:
            input_dim: Input dimension (ML logits + LLM scores)
            output_dim: Output dimension (number of classes)
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fusion MLP."""
        return self.network(x)


class FusionWrapper(nn.Module):
    """Wrapper that combines ML model with frozen LLM scores via Fusion MLP."""
    
    def __init__(self, ml_model, num_labels: int, task: str = "multiclass", 
                 hidden_dims: List[int] = [64, 32]):
        """Initialize Fusion Wrapper.
        
        Args:
            ml_model: Pre-trained ML model (e.g., RoBERTa)
            num_labels: Number of output labels
            task: "multiclass" or "multilabel"
            hidden_dims: Hidden dimensions for fusion MLP
        """
        super().__init__()
        self.ml_model = ml_model
        self.num_labels = num_labels
        self.task = task
        
        # Fusion MLP takes ML logits + LLM scores
        fusion_input_dim = num_labels * 2  # ML logits + LLM scores
        self.fusion_mlp = FusionMLP(fusion_input_dim, num_labels, hidden_dims)
        
        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                llm_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass combining ML and LLM predictions.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            llm_scores: Pre-computed LLM scores (detached)
            
        Returns:
            Dict containing ML logits, LLM scores, and fused logits
        """
        # Get ML logits
        ml_outputs = self.ml_model.model(input_ids=input_ids, attention_mask=attention_mask)
        ml_logits = ml_outputs.logits
        
        # Ensure LLM scores are detached (no gradient flow)
        llm_scores = llm_scores.detach()
        
        # Concatenate ML logits and LLM scores
        fusion_input = torch.cat([ml_logits, llm_scores], dim=1)
        
        # Generate fused predictions
        fused_logits = self.fusion_mlp(fusion_input)
        
        return {
            "ml_logits": ml_logits,
            "llm_scores": llm_scores,
            "fused_logits": fused_logits
        }


class FusionEnsemble(BaseEnsemble):
    """Ensemble that fuses ML and LLM classifiers with trainable MLP."""
    
    def __init__(self, ensemble_config):
        """Initialize fusion ensemble.
        
        Args:
            ensemble_config: Configuration for the ensemble
        """
        super().__init__(ensemble_config)
        
        # Fusion-specific parameters
        self.fusion_hidden_dims = ensemble_config.parameters.get('fusion_hidden_dims', [64, 32])
        self.ml_lr = ensemble_config.parameters.get('ml_lr', 1e-5)  # Small LR for ML backbone
        self.fusion_lr = ensemble_config.parameters.get('fusion_lr', 1e-3)  # Larger LR for fusion MLP
        self.num_epochs = ensemble_config.parameters.get('num_epochs', 10)
        self.batch_size = ensemble_config.parameters.get('batch_size', 16)
        
        # Model components
        self.ml_model = None
        self.llm_model = None
        self.fusion_wrapper = None
        self.llm_scores_cache = {}
        self.calibrator = None
        self.test_performance = {}  # Store test set performance
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def add_ml_model(self, ml_model):
        """Add ML model (e.g., RoBERTa) to the fusion ensemble."""
        self.ml_model = ml_model
        self.models.append(ml_model)
        self.model_names.append("ml_model")
    
    def add_llm_model(self, llm_model):
        """Add LLM model to the fusion ensemble."""
        self.llm_model = llm_model
        self.models.append(llm_model)
        self.model_names.append("llm_model")
    
    def fit(self, training_data: TrainingData) -> None:
        """Train the fusion ensemble with proper train/val/test split.
        
        Args:
            training_data: Training data containing texts and labels
        """
        if self.ml_model is None or self.llm_model is None:
            raise EnsembleError("Both ML and LLM models must be added before training")
        
        self.classification_type = training_data.classification_type
        
        # Step 1: Split data into train/val/test (60/20/20)
        print("Splitting data into train/validation/test sets...")
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            training_data.texts, training_data.labels,
            test_size=0.4, random_state=42, stratify=None
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=0.5, random_state=42, stratify=None
        )
        
        print(f"   📊 Train: {len(train_texts)} samples")
        print(f"   📊 Validation: {len(val_texts)} samples") 
        print(f"   📊 Test: {len(test_texts)} samples")
        
        # Step 2: Train ML model on training set only
        if not self.ml_model.is_trained:
            print("Training ML model on training set...")
            train_data = TrainingData(
                texts=train_texts,
                labels=train_labels,
                classification_type=self.classification_type
            )
            self.ml_model.fit(train_data)
        else:
            print("ML model already trained")
        
        # Set up classes from ML model
        self.classes_ = self.ml_model.classes_
        self.num_labels = len(self.classes_)
        
        # Optional: Get LLM predictions on training set (for analysis/comparison)
        # Note: We don't use these for training to avoid data leakage
        print("Getting LLM predictions on training set (for reference)...")
        train_llm_scores = self._get_llm_scores(train_texts, data_split="training")
        
        # Step 3: Get ML predictions on validation set
        print("Getting ML predictions on validation set...")
        ml_val_result = self.ml_model.predict(val_texts)
        
        # Step 4: Get LLM predictions on validation set
        print("Getting LLM predictions on validation set...")
        val_llm_scores = self._get_llm_scores(val_texts, data_split="validation")
        
        # Step 5: Create fusion wrapper
        print("Creating fusion wrapper...")
        task = "multilabel" if self.classification_type == ClassificationType.MULTI_LABEL else "multiclass"
        self.fusion_wrapper = FusionWrapper(
            ml_model=self.ml_model,
            num_labels=self.num_labels,
            task=task,
            hidden_dims=self.fusion_hidden_dims
        )
        
        # Step 6: Train fusion MLP on validation set predictions
        print("Training fusion MLP on validation predictions...")
        self._train_fusion_mlp_on_val(val_texts, val_labels, val_llm_scores)
        
        # Step 7: Calibrate LLM scores
        print("Calibrating LLM scores...")
        self._calibrate_llm_scores(val_llm_scores, val_labels)
        
        # Step 8: Evaluate on test set
        print("Evaluating ensemble on test set...")
        test_result = self.predict(test_texts, test_labels)
        
        # Store test performance
        self.test_performance = test_result.metadata.get('metrics', {}) if test_result.metadata else {}
        print(f"   📈 Test Performance: {self.test_performance}")
        
        self.is_trained = True
        print("✅ Fusion ensemble training completed!")
    
    def _train_fusion_mlp_on_val(self, val_texts: List[str], val_labels: List[List[int]], 
                                val_llm_scores: np.ndarray):
        """Train the fusion MLP using validation set predictions."""
        # Split validation set into train/val for fusion MLP training
        fusion_train_texts, fusion_val_texts, fusion_train_labels, fusion_val_labels, \
        fusion_train_llm_scores, fusion_val_llm_scores = train_test_split(
            val_texts, val_labels, val_llm_scores,
            test_size=0.3, random_state=42, stratify=None
        )
        
        print(f"   🔧 Fusion training: {len(fusion_train_texts)} samples")
        print(f"   🔧 Fusion validation: {len(fusion_val_texts)} samples")
        
        # Create data loaders
        train_dataset = self._create_fusion_dataset(fusion_train_texts, fusion_train_labels, fusion_train_llm_scores)
        val_dataset = self._create_fusion_dataset(fusion_val_texts, fusion_val_labels, fusion_val_llm_scores)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Setup optimizer with different learning rates
        ml_params = list(self.fusion_wrapper.ml_model.parameters())
        fusion_params = list(self.fusion_wrapper.fusion_mlp.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': ml_params, 'lr': self.ml_lr},
            {'params': fusion_params, 'lr': self.fusion_lr}
        ])
        
        # Loss function
        if self.classification_type == ClassificationType.MULTI_CLASS:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Training loop with validation monitoring
        self.fusion_wrapper.train()
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training phase
            total_train_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                llm_scores = batch['llm_scores'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.fusion_wrapper(input_ids, attention_mask, llm_scores)
                fused_logits = outputs['fused_logits']
                
                if self.classification_type == ClassificationType.MULTI_CLASS:
                    labels = torch.argmax(labels, dim=1)
                
                loss = criterion(fused_logits, labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            # Validation phase
            self.fusion_wrapper.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    llm_scores = batch['llm_scores'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.fusion_wrapper(input_ids, attention_mask, llm_scores)
                    fused_logits = outputs['fused_logits']
                    
                    if self.classification_type == ClassificationType.MULTI_CLASS:
                        labels = torch.argmax(labels, dim=1)
                    
                    loss = criterion(fused_logits, labels)
                    total_val_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f"   Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Could save model state here if needed
            
            self.fusion_wrapper.train()  # Back to training mode
    
    def _get_llm_scores(self, texts: List[str], data_split: str = "unknown") -> np.ndarray:
        """Get LLM scores using existing LLM model.
        
        Args:
            texts: List of texts to get predictions for
            data_split: Which data split this is ("training", "validation", "test", "inference")
                       Used for logging and caching purposes
        
        Returns:
            numpy array of LLM scores/predictions
        """
        # Cache key based on texts and data split
        cache_key = hash(tuple(texts + [data_split]))
        
        if cache_key in self.llm_scores_cache:
            print(f"   📋 Using cached LLM scores for {data_split} set ({len(texts)} samples)")
            return self.llm_scores_cache[cache_key]
        
        print(f"   🧠 Generating LLM predictions for {data_split} set ({len(texts)} samples)...")
        
        # Use existing LLM model to get predictions
        # Convert LLM predictions to scores format
        llm_result = self.llm_model.predict(texts)
        
        # Convert LLM predictions to binary scores
        scores = np.zeros((len(texts), self.num_labels))
        
        for i, prediction in enumerate(llm_result.predictions):
            if isinstance(prediction, str):
                # Multi-class: single prediction
                if prediction in self.classes_:
                    class_idx = self.classes_.index(prediction)
                    scores[i, class_idx] = 1.0
            else:
                # Multi-label: list of predictions
                for pred_class in prediction:
                    if pred_class in self.classes_:
                        class_idx = self.classes_.index(pred_class)
                        scores[i, class_idx] = 1.0
        
        # Use confidence scores if available
        if llm_result.confidence_scores:
            for i, confidence in enumerate(llm_result.confidence_scores):
                scores[i] *= confidence
        
        # Cache the results
        self.llm_scores_cache[cache_key] = scores
        print(f"   ✅ Generated and cached LLM scores for {data_split} set")
        
        return scores
    
    def _create_fusion_dataset(self, texts: List[str], labels: List[List[int]], 
                              llm_scores: np.ndarray):
        """Create dataset for fusion training."""
        # Tokenize texts using ML model's tokenizer
        tokenized = []
        for text in texts:
            encoding = self.ml_model.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.ml_model.max_length,
                return_tensors='pt'
            )
            tokenized.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })
        
        # Create tensor dataset
        input_ids = torch.stack([item['input_ids'] for item in tokenized])
        attention_mask = torch.stack([item['attention_mask'] for item in tokenized])
        llm_scores_tensor = torch.FloatTensor(llm_scores)
        labels_tensor = torch.FloatTensor(labels)
        
        return torch.utils.data.TensorDataset(input_ids, attention_mask, llm_scores_tensor, labels_tensor)
    
    def _calibrate_llm_scores(self, val_llm_scores: np.ndarray, val_labels: List[List[int]]):
        """Calibrate LLM scores using validation data."""
        if self.classification_type == ClassificationType.MULTI_CLASS:
            # Use temperature scaling for multi-class
            from sklearn.linear_model import LogisticRegression
            val_labels_indices = np.argmax(val_labels, axis=1)
            self.calibrator = LogisticRegression()
            self.calibrator.fit(val_llm_scores, val_labels_indices)
        else:
            # Use isotonic regression for multi-label
            self.calibrator = {}
            for i in range(self.num_labels):
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                iso_reg.fit(val_llm_scores[:, i], np.array(val_labels)[:, i])
                self.calibrator[i] = iso_reg
    
    def predict(self, texts: List[str], true_labels: Optional[List[List[int]]] = None) -> ClassificationResult:
        """Predict using fusion ensemble."""
        if not self.is_trained:
            raise EnsembleError("Fusion ensemble must be trained before prediction")
        
        # Determine data split based on whether true_labels are provided
        data_split = "test" if true_labels is not None else "inference"
        
        # Get LLM scores
        llm_scores = self._get_llm_scores(texts, data_split=data_split)
        
        # Calibrate LLM scores
        if self.calibrator:
            if self.classification_type == ClassificationType.MULTI_CLASS:
                llm_scores = self.calibrator.predict_proba(llm_scores)
            else:
                for i in range(self.num_labels):
                    llm_scores[:, i] = self.calibrator[i].transform(llm_scores[:, i])
        
        # Create dataset
        dataset = self._create_fusion_dataset(texts, [[0] * self.num_labels] * len(texts), llm_scores)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Generate predictions
        self.fusion_wrapper.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, llm_scores_batch, _ = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                llm_scores_batch = llm_scores_batch.to(self.device)
                
                outputs = self.fusion_wrapper(input_ids, attention_mask, llm_scores_batch)
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
