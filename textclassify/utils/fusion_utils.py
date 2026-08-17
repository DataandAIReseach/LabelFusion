"""
Utility functions and helpers for LLM+ML Fusion models.

This module provides utility functions for:
- Data preparation and validation
- Model loading and saving
- Evaluation and metrics
- Visualization and reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import pickle
import yaml

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    PLOTTING_AVAILABLE = False

from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.core.types import ClassificationResult, ClassificationType


class FusionUtils:
    """Utility class for fusion ensemble operations."""
    
    @staticmethod
    def prepare_data_for_fusion(
        df: pd.DataFrame,
        text_column: str,
        label_columns: List[str],
        multi_label: bool = False,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare and split data for fusion training.
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            label_columns: Names of label columns
            multi_label: Whether this is multi-label classification
            test_size: Fraction for test split
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # Validate columns exist
        missing_cols = [col for col in [text_column] + label_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataframe: {missing_cols}")
        
        # Handle missing values
        df = df.dropna(subset=[text_column])
        
        # Stratification for splitting
        if multi_label:
            # For multi-label, we can't use simple stratification
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
        else:
            # For single-label, use stratification
            if len(label_columns) == 1:
                stratify = df[label_columns[0]]
            else:
                # Create combined label for stratification
                stratify = df[label_columns].apply(lambda x: '|'.join(x.astype(str)), axis=1)
            
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=stratify
            )
        
        return train_df, test_df
    
    @staticmethod
    def validate_fusion_data(
        df: pd.DataFrame,
        text_column: str,
        label_columns: List[str],
        multi_label: bool = False
    ) -> Dict[str, Any]:
        """Validate data for fusion training and return summary statistics.
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            label_columns: Names of label columns
            multi_label: Whether this is multi-label classification
            
        Returns:
            Dictionary with validation results and statistics
        """
        stats = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check basic requirements
        if text_column not in df.columns:
            stats['errors'].append(f"Text column '{text_column}' not found")
            stats['valid'] = False
        
        missing_labels = [col for col in label_columns if col not in df.columns]
        if missing_labels:
            stats['errors'].append(f"Label columns not found: {missing_labels}")
            stats['valid'] = False
        
        if not stats['valid']:
            return stats
        
        # Check for missing values
        text_missing = df[text_column].isna().sum()
        if text_missing > 0:
            stats['warnings'].append(f"{text_missing} missing text values found")
        
        label_missing = df[label_columns].isna().sum().sum()
        if label_missing > 0:
            stats['warnings'].append(f"{label_missing} missing label values found")
        
        # Text statistics
        text_lengths = df[text_column].str.len()
        stats['statistics']['text'] = {
            'count': len(df),
            'avg_length': text_lengths.mean(),
            'min_length': text_lengths.min(),
            'max_length': text_lengths.max(),
            'empty_texts': (text_lengths == 0).sum()
        }
        
        # Label statistics
        if multi_label:
            # Multi-label statistics
            label_sums = df[label_columns].sum()
            stats['statistics']['labels'] = {
                'num_labels': len(label_columns),
                'label_distribution': label_sums.to_dict(),
                'avg_labels_per_sample': df[label_columns].sum(axis=1).mean(),
                'samples_with_no_labels': (df[label_columns].sum(axis=1) == 0).sum()
            }
        else:
            # Single-label statistics
            if len(label_columns) == 1:
                value_counts = df[label_columns[0]].value_counts()
                stats['statistics']['labels'] = {
                    'num_classes': len(value_counts),
                    'class_distribution': value_counts.to_dict(),
                    'most_common_class': value_counts.index[0],
                    'least_common_class': value_counts.index[-1]
                }
            else:
                # Binary encoded single-label
                label_sums = df[label_columns].sum()
                stats['statistics']['labels'] = {
                    'num_classes': len(label_columns),
                    'class_distribution': label_sums.to_dict()
                }
        
        return stats
    
    @staticmethod
    def load_fusion_model(model_path: str) -> FusionEnsemble:
        """Load a trained fusion ensemble from disk.
        
        Args:
            model_path: Path to saved model directory or pickle file
            
        Returns:
            Loaded FusionEnsemble
        """
        model_path = Path(model_path)
        
        if model_path.is_dir():
            # Load from directory
            pickle_path = model_path / 'fusion_ensemble.pkl'
            if not pickle_path.exists():
                raise FileNotFoundError(f"fusion_ensemble.pkl not found in {model_path}")
        else:
            # Load from pickle file
            pickle_path = model_path
        
        with open(pickle_path, 'rb') as f:
            fusion_ensemble = pickle.load(f)
        
        return fusion_ensemble
    
    @staticmethod
    def save_fusion_model(
        fusion_ensemble: FusionEnsemble,
        save_path: str,
        include_config: bool = True
    ):
        """Save a trained fusion ensemble to disk.
        
        Args:
            fusion_ensemble: Trained fusion ensemble
            save_path: Path to save directory
            include_config: Whether to save configuration files
        """
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Save main model
        model_path = save_path / 'fusion_ensemble.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(fusion_ensemble, f)
        
        if include_config:
            # Save model info
            model_info = {
                'classification_type': fusion_ensemble.classification_type.value,
                'num_labels': fusion_ensemble.num_labels,
                'classes': fusion_ensemble.classes_,
                'ml_model_info': fusion_ensemble.ml_model.model_info if fusion_ensemble.ml_model else None,
                'llm_model_info': fusion_ensemble.llm_model.model_info if fusion_ensemble.llm_model else None
            }
            
            info_path = save_path / 'model_info.yaml'
            with open(info_path, 'w') as f:
                yaml.dump(model_info, f, default_flow_style=False)
    
    @staticmethod
    def evaluate_fusion_model(
        fusion_ensemble: FusionEnsemble,
        test_texts: List[str],
        test_labels: List[List[int]],
        save_results: bool = False,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of fusion model.
        
        Args:
            fusion_ensemble: Trained fusion ensemble
            test_texts: Test texts
            test_labels: True labels
            save_results: Whether to save evaluation results
            save_path: Path to save results
            
        Returns:
            Dictionary with evaluation results
        """
        # Get predictions
        result = fusion_ensemble.predict(test_texts, test_labels)
        
        # Extract metrics
        evaluation = {
            'predictions': result.predictions,
            'true_labels': test_labels,
            'metrics': result.metadata.get('metrics', {}) if result.metadata else {},
            'model_info': {
                'classification_type': fusion_ensemble.classification_type.value,
                'num_labels': fusion_ensemble.num_labels,
                'classes': fusion_ensemble.classes_
            }
        }
        
        # Add detailed analysis
        evaluation['analysis'] = FusionUtils._analyze_predictions(
            result.predictions, test_labels, fusion_ensemble.classes_, fusion_ensemble.classification_type
        )
        
        # Save results if requested
        if save_results and save_path:
            save_path = Path(save_path)
            save_path.mkdir(exist_ok=True, parents=True)
            
            results_path = save_path / 'evaluation_results.yaml'
            with open(results_path, 'w') as f:
                yaml.dump(evaluation, f, default_flow_style=False)
        
        return evaluation
    
    @staticmethod
    def _analyze_predictions(
        predictions: List[Union[str, List[str]]],
        true_labels: List[List[int]],
        classes: List[str],
        classification_type: ClassificationType
    ) -> Dict[str, Any]:
        """Analyze predictions in detail."""
        analysis = {}
        
        if classification_type == ClassificationType.MULTI_CLASS:
            # Convert predictions and true labels to indices
            pred_indices = []
            true_indices = []
            
            for pred, true_label in zip(predictions, true_labels):
                if isinstance(pred, str) and pred in classes:
                    pred_indices.append(classes.index(pred))
                else:
                    pred_indices.append(-1)  # Unknown prediction
                
                true_indices.append(np.argmax(true_label))
            
            # Per-class analysis
            from sklearn.metrics import classification_report, confusion_matrix
            
            report = classification_report(
                true_indices, pred_indices, 
                target_names=classes, output_dict=True, zero_division=0
            )
            analysis['classification_report'] = report
            
            # Confusion matrix
            cm = confusion_matrix(true_indices, pred_indices)
            analysis['confusion_matrix'] = cm.tolist()
            
        else:
            # Multi-label analysis
            # Convert predictions to binary format
            pred_binary = []
            for pred in predictions:
                binary = [0] * len(classes)
                if isinstance(pred, list):
                    for class_name in pred:
                        if class_name in classes:
                            binary[classes.index(class_name)] = 1
                pred_binary.append(binary)
            
            # Per-label metrics
            from sklearn.metrics import classification_report
            
            pred_array = np.array(pred_binary)
            true_array = np.array(true_labels)
            
            analysis['per_label_metrics'] = {}
            for i, class_name in enumerate(classes):
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, support = precision_recall_fscore_support(
                    true_array[:, i], pred_array[:, i], average='binary', zero_division=0
                )
                analysis['per_label_metrics'][class_name] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'support': int(support)
                }
        
        return analysis
    
    @staticmethod
    def plot_fusion_results(
        evaluation: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Create visualizations for fusion model results.
        
        Args:
            evaluation: Evaluation results from evaluate_fusion_model
            save_path: Path to save plots
            show: Whether to display plots
        """
        classification_type = evaluation['model_info']['classification_type']
        
        if classification_type == 'multi_class':
            FusionUtils._plot_multiclass_results(evaluation, save_path, show)
        else:
            FusionUtils._plot_multilabel_results(evaluation, save_path, show)
    
    @staticmethod
    def _plot_multiclass_results(evaluation: Dict[str, Any], save_path: Optional[str], show: bool):
        """Plot multi-class classification results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        classes = evaluation['model_info']['classes']
        cm = np.array(evaluation['analysis']['confusion_matrix'])
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # Per-class F1 scores
        report = evaluation['analysis']['classification_report']
        f1_scores = [report[cls]['f1-score'] for cls in classes if cls in report]
        axes[0, 1].bar(classes, f1_scores)
        axes[0, 1].set_title('Per-Class F1 Scores')
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Metrics summary
        metrics = evaluation['metrics']
        metric_names = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        metric_values = [metrics.get(name, 0) for name in metric_names]
        
        axes[1, 0].bar(metric_names, metric_values)
        axes[1, 0].set_title('Overall Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Class distribution
        support_values = [report[cls]['support'] for cls in classes if cls in report]
        axes[1, 1].pie(support_values, labels=classes, autopct='%1.1f%%')
        axes[1, 1].set_title('Class Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/multiclass_results.png", dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def _plot_multilabel_results(evaluation: Dict[str, Any], save_path: Optional[str], show: bool):
        """Plot multi-label classification results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        classes = evaluation['model_info']['classes']
        per_label = evaluation['analysis']['per_label_metrics']
        
        # Per-label F1 scores
        f1_scores = [per_label[cls]['f1'] for cls in classes]
        axes[0, 0].bar(classes, f1_scores)
        axes[0, 0].set_title('Per-Label F1 Scores')
        axes[0, 0].set_xlabel('Labels')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Per-label Precision
        precision_scores = [per_label[cls]['precision'] for cls in classes]
        axes[0, 1].bar(classes, precision_scores)
        axes[0, 1].set_title('Per-Label Precision')
        axes[0, 1].set_xlabel('Labels')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Per-label Recall
        recall_scores = [per_label[cls]['recall'] for cls in classes]
        axes[1, 0].bar(classes, recall_scores)
        axes[1, 0].set_title('Per-Label Recall')
        axes[1, 0].set_xlabel('Labels')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall metrics
        metrics = evaluation['metrics']
        metric_names = ['exact_match_accuracy', 'hamming_loss', 'f1_weighted']
        metric_values = [metrics.get(name, 0) for name in metric_names]
        
        axes[1, 1].bar(metric_names, metric_values)
        axes[1, 1].set_title('Overall Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/multilabel_results.png", dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


# Convenience functions for quick access
def prepare_fusion_data(df, text_col, label_cols, multi_label=False, test_size=0.2):
    """Quick data preparation for fusion training."""
    return FusionUtils.prepare_data_for_fusion(df, text_col, label_cols, multi_label, test_size)

def validate_fusion_data(df, text_col, label_cols, multi_label=False):
    """Quick data validation for fusion training."""
    return FusionUtils.validate_fusion_data(df, text_col, label_cols, multi_label)

def load_fusion_model(path):
    """Quick model loading."""
    return FusionUtils.load_fusion_model(path)

def evaluate_fusion_model(model, texts, labels, save_results=False, save_path=None):
    """Quick model evaluation."""
    return FusionUtils.evaluate_fusion_model(model, texts, labels, save_results, save_path)
