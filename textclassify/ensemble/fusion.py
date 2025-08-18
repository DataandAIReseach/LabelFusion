"""Fusion ensemble combining ML and LLM classifiers with trainable MLP."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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
    
    def forward(self, ml_predictions: torch.Tensor, llm_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass combining ML and LLM predictions.
        
        Args:
            ml_predictions: Pre-computed ML predictions/logits
            llm_predictions: Pre-computed LLM predictions/scores
            
        Returns:
            Dict containing ML predictions, LLM predictions, and fused logits
        """
        # Ensure predictions are detached (no gradient flow to original models)
        ml_predictions = ml_predictions.detach()
        llm_predictions = llm_predictions.detach()
        
        # Concatenate ML and LLM predictions
        fusion_input = torch.cat([ml_predictions, llm_predictions], dim=1)
        
        # Generate fused predictions through MLP
        fused_logits = self.fusion_mlp(fusion_input)
        
        return {
            "ml_predictions": ml_predictions,
            "llm_predictions": llm_predictions,
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
        
        # Initialize training state
        self.is_trained = False
        
        # Determine classification type from ensemble config or infer from models
        if 'classification_type' in ensemble_config.parameters:
            self.classification_type = ensemble_config.parameters['classification_type']
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
        """Add LLM model to the fusion ensemble."""
        self.llm_model = llm_model
        self.models.append(llm_model)
        self.model_names.append("llm_model")
    
    def fit(self, df: Union[TrainingData, pd.DataFrame, Dict]) -> None:
        """Train the fusion ensemble with proper train/val/test split.
        
        Args:
            df: Training data in various formats (DataFrame, TrainingData, or Dict)
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
        
        # Work with DataFrame directly as long as possible
        if isinstance(df, TrainingData):
            # Convert TrainingData to DataFrame for consistent processing
            labels = df.labels
            label_data = {}
            # Create label columns from the labels
            if labels and isinstance(labels[0], list):
                # 2D format: [[1,0,0], [0,1,0], ...]
                for i in range(len(labels[0])):
                    label_data[f'label_{i}'] = [label[i] for label in labels]
            else:
                # This shouldn't happen with TrainingData but handle gracefully
                raise EnsembleError("TrainingData labels should be in 2D list format")
            
            working_df = pd.DataFrame({
                'text': df.texts,
                **label_data
            })
            text_column = 'text'
            label_columns = [col for col in working_df.columns if col.startswith('label_')]
        elif isinstance(df, pd.DataFrame):
            working_df = df.copy()
            text_column = 'text'  # Could be configurable
            if text_column not in working_df.columns:
                raise EnsembleError(f"Text column '{text_column}' not found in DataFrame")
            label_columns = [col for col in working_df.columns if col != text_column]
        elif isinstance(df, dict):
            # Convert dict to DataFrame for consistent processing
            labels = df['labels']
            label_data = {}
            # Handle both 2D list format and 1D list format
            if labels and isinstance(labels[0], list):
                # 2D format: [[1,0,0], [0,1,0], ...]
                for i in range(len(labels[0])):
                    label_data[f'label_{i}'] = [label[i] for label in labels]
            else:
                # 1D format: [0, 1, 2, ...] - convert to one-hot
                unique_labels = sorted(set(labels))
                for i, unique_label in enumerate(unique_labels):
                    label_data[f'label_{i}'] = [1 if label == unique_label else 0 for label in labels]
            
            working_df = pd.DataFrame({
                'text': df['texts'],
                **label_data
            })
            text_column = 'text'
            label_columns = [col for col in working_df.columns if col.startswith('label_')]
        else:
            raise EnsembleError(f"Unsupported training data format: {type(df)}")
        
        print(f"Classification type: {self.classification_type}")
        print(f"Training data: {len(working_df)} samples")
        
        # Step 1: Split DataFrames directly (60/20/20)
        print("Splitting data into train/validation/test sets...")
        train_df, temp_df = train_test_split(
            working_df, 
            test_size=0.4, 
            random_state=42, 
            stratify=None
        )
        
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            random_state=42, 
            stratify=None
        )
        
        print(f"   📊 Train: {len(train_df)} samples")
        print(f"   📊 Validation: {len(val_df)} samples") 
        print(f"   📊 Test: {len(test_df)} samples")
        
        # Step 2: Train ML model on training set only
        if not self.ml_model.is_trained:
            print("Training ML model on training set...")
            # Convert to TrainingData format only when needed for ML model
            train_data = TrainingData(
                texts=train_df[text_column].tolist(),
                labels=train_df[label_columns].values.tolist(),
                classification_type=self.classification_type
            )
            self.ml_model.fit(train_data)
        else:
            print("ML model already trained")
        
        # Set up classes from ML model
        self.classes_ = self.ml_model.classes_
        self.num_labels = len(self.classes_)
        
        # Step 3: Get ML predictions on validation set
        print("Getting ML predictions on validation set...")
        ml_val_result = self.ml_model.predict(texts=val_df[text_column].tolist(), 
                                             true_labels=val_df[label_columns].values.tolist())
        
    

        # Step 4: Get LLM predictions on validation set
        print("Getting LLM predictions on validation set...")
        
        # Use DataFrames directly for LLM model - no need to recreate them
        val_llm_result = self.llm_model.predict(train_df=train_df,
                                                test_df=val_df)
        
        
        
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
        self._train_fusion_mlp_on_val(val_df, ml_val_result, val_llm_result, text_column, label_columns)
        
        # Step 7: Store LLM result and training data for later use
        print("Storing LLM validation results and training data...")
        self.val_llm_result = val_llm_result
        self.train_df_cache = train_df.copy()  # Cache training DataFrame for LLM predictions
        
        # Step 8: Evaluate on test set
        print("Evaluating ensemble on test set...")
        
        # Set training flag early to avoid recursive prediction issues
        self.is_trained = True
        
        try:
            test_result = self.predict(test_df)
            
            # Store test performance
            self.test_performance = test_result.metadata.get('metrics', {}) if test_result.metadata else {}
            print(f"   📈 Test Performance: {self.test_performance}")
        except Exception as e:
            print(f"   ⚠️ Warning: Test evaluation failed: {e}")
            self.test_performance = {}
        
        print("✅ Fusion ensemble training completed!")
    
    def _train_fusion_mlp_on_val(self, val_df: pd.DataFrame, ml_val_result, val_llm_result, 
                                text_column: str, label_columns: List[str]):
        """Train the fusion MLP using validation set predictions from both ML and LLM models."""
        
        # Split validation DataFrame into train/val for fusion MLP training
        fusion_train_df, fusion_val_df = train_test_split(
            val_df,
            test_size=0.1, random_state=42, stratify=None
        )
        
        # Split both ML and LLM results accordingly
        split_idx = len(fusion_train_df)
        
        # Split ML predictions
        fusion_train_ml_predictions = ml_val_result.predictions[:split_idx]
        fusion_val_ml_predictions = ml_val_result.predictions[split_idx:]
        
        # Split LLM predictions
        fusion_train_llm_predictions = val_llm_result.predictions[:split_idx]
        fusion_val_llm_predictions = val_llm_result.predictions[split_idx:]
        
        print(f"   🔧 Fusion training: {len(fusion_train_df)} samples")
        print(f"   🔧 Fusion validation: {len(fusion_val_df)} samples")
        
        # Create data loaders using both ML and LLM predictions
        train_dataset = self._create_fusion_dataset(
            fusion_train_df[text_column].tolist(), 
            fusion_train_df[label_columns].values.tolist(), 
            fusion_train_ml_predictions,
            fusion_train_llm_predictions
        )
        val_dataset = self._create_fusion_dataset(
            fusion_val_df[text_column].tolist(), 
            fusion_val_df[label_columns].values.tolist(), 
            fusion_val_ml_predictions,
            fusion_val_llm_predictions
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Freeze ML model parameters - only optimize the fusion MLP
        for param in self.fusion_wrapper.ml_model.model.parameters():
            param.requires_grad = False
        
        # Setup optimizer for fusion MLP only
        fusion_params = list(self.fusion_wrapper.fusion_mlp.parameters())
        optimizer = torch.optim.AdamW(fusion_params, lr=self.fusion_lr)
        
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
                ml_predictions, llm_predictions, labels = batch
                ml_predictions = ml_predictions.to(self.device)
                llm_predictions = llm_predictions.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.fusion_wrapper(ml_predictions, llm_predictions)
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
                    ml_predictions, llm_predictions, labels = batch
                    ml_predictions = ml_predictions.to(self.device)
                    llm_predictions = llm_predictions.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.fusion_wrapper(ml_predictions, llm_predictions)
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
    
    def _create_fusion_dataset(self, texts: List[str], labels: List[List[int]], 
                              ml_predictions: List, llm_predictions: List):
        """Create dataset for fusion training using pre-computed ML and LLM predictions."""
        
        # Convert ML predictions to tensor format (binary vectors)
        ml_tensor = torch.zeros(len(ml_predictions), self.num_labels)
        for i, prediction in enumerate(ml_predictions):
            if isinstance(prediction, list) and len(prediction) == self.num_labels:
                # Already in binary vector format
                ml_tensor[i] = torch.tensor(prediction, dtype=torch.float)
            elif isinstance(prediction, str):
                # Convert class name to binary vector
                if prediction in self.classes_:
                    class_idx = self.classes_.index(prediction)
                    ml_tensor[i, class_idx] = 1.0
        
        # Convert LLM predictions to tensor format (binary vectors)
        llm_tensor = torch.zeros(len(llm_predictions), self.num_labels)
        for i, prediction in enumerate(llm_predictions):
            if isinstance(prediction, list) and len(prediction) == self.num_labels:
                # Already in binary vector format
                llm_tensor[i] = torch.tensor(prediction, dtype=torch.float)
            elif isinstance(prediction, str):
                # Convert class name to binary vector
                if prediction in self.classes_:
                    class_idx = self.classes_.index(prediction)
                    llm_tensor[i, class_idx] = 1.0
            elif isinstance(prediction, list):
                # List of class names (multi-label)
                for pred_class in prediction:
                    if pred_class in self.classes_:
                        class_idx = self.classes_.index(pred_class)
                        llm_tensor[i, class_idx] = 1.0
        
        # Create tensor dataset with predictions only (no tokenization needed)
        labels_tensor = torch.FloatTensor(labels)
        
        return torch.utils.data.TensorDataset(ml_tensor, llm_tensor, labels_tensor)
    
    def predict(self, data: pd.DataFrame) -> ClassificationResult:
        """Predict using fusion ensemble.
        
        Args:
            data: DataFrame with text column (and optionally label columns)
        """
        if not self.is_trained:
            raise EnsembleError("Fusion ensemble must be trained before prediction")
        
        # DataFrame input only
        test_df = data.copy()
        text_column = 'text'  # Could be configurable
        if text_column not in test_df.columns:
            raise EnsembleError(f"Text column '{text_column}' not found in DataFrame")
        texts = test_df[text_column].tolist()
        
        # Extract true labels from DataFrame if available
        extracted_labels = None
        if hasattr(self, 'classes_') and self.classes_:
            label_columns = [col for col in self.classes_ if col in test_df.columns]
            if label_columns:
                extracted_labels = test_df[label_columns].values.tolist()
        
        # Get ML predictions - match the format used in fit()
        ml_result = self.ml_model.predict(texts=texts)
        
        # Get LLM predictions - match the format used in fit()
        llm_result = self.llm_model.predict(train_df=self.train_df_cache, test_df=test_df)
        
        # Create dataset using both ML and LLM predictions
        dummy_labels = [[0] * self.num_labels] * len(texts)
        dataset = self._create_fusion_dataset(texts, dummy_labels, ml_result.predictions, llm_result.predictions)
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
        
        return self._create_result(predictions=all_predictions, true_labels=extracted_labels)
    
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
