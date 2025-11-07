"""Centralized results and artifacts management following ML best practices."""

import os
import json
import yaml
import pickle
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass, asdict
import logging

from ..core.types import ClassificationResult, ClassificationType
from ..core.exceptions import PersistenceError


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment run."""
    experiment_id: str
    timestamp: str
    model_name: str
    model_type: str
    dataset_hash: str
    config: Dict[str, Any]
    git_commit: Optional[str] = None
    environment: Optional[Dict[str, str]] = None


class ResultsManager:
    """Centralized manager for saving predictions, models, and experiment artifacts."""
    
    def __init__(self, 
                 base_output_dir: str = "outputs",
                 experiment_name: Optional[str] = None,
                 auto_create_dirs: bool = True):
        """Initialize the results manager.
        
        Args:
            base_output_dir: Base directory for all outputs
            experiment_name: Name of the current experiment
            auto_create_dirs: Whether to automatically create directories
        """
        self.base_output_dir = Path(base_output_dir)
        self.auto_create_dirs = auto_create_dirs
        
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_id = f"{timestamp}_{experiment_name}" if experiment_name else timestamp
        
        # Define directory structure
        self.experiment_dir = self.base_output_dir / "experiments" / self.experiment_id
        self.predictions_dir = self.experiment_dir / "predictions"
        self.models_dir = self.experiment_dir / "models"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.logs_dir = self.experiment_dir / "logs"
        self.plots_dir = self.experiment_dir / "plots"
        self.cache_dir = self.base_output_dir / "cache"
        
        # Create directories
        if self.auto_create_dirs:
            self._create_directories()
        
        # Setup logging
        self.logger = self._setup_experiment_logging()
        
        # Initialize metadata
        self.metadata = ExperimentMetadata(
            experiment_id=self.experiment_id,
            timestamp=datetime.now().isoformat(),
            model_name="",
            model_type="",
            dataset_hash="",
            config={},
            git_commit=self._get_git_commit(),
            environment=self._get_environment_info()
        )
    
    def _create_directories(self):
        """Create the standard directory structure."""
        dirs_to_create = [
            self.experiment_dir,
            self.predictions_dir,
            self.models_dir,
            self.metrics_dir,
            self.logs_dir,
            self.plots_dir,
            self.cache_dir
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_experiment_logging(self) -> logging.Logger:
        """Setup logging for the experiment."""
        logger = logging.getLogger(f"experiment_{self.experiment_id}")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.logs_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler if not already added
        if not logger.handlers:
            logger.addHandler(file_handler)
        
        return logger
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get environment information."""
        import sys
        import platform
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "working_directory": str(Path.cwd())
        }
    
    def _create_dataset_hash(self, df: pd.DataFrame) -> str:
        """Create a hash of the DataFrame for versioning."""
        df_hash = hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()[:12]  # 12 characters for readability
        return df_hash
    
    def save_predictions(self, 
                        result: ClassificationResult,
                        dataset_name: str,
                        dataset_df: Optional[pd.DataFrame] = None,
                        format: str = "both") -> Dict[str, str]:
        """Save predictions in multiple formats.
        
        Args:
            result: Classification result to save
            dataset_name: Name of the dataset (train/val/test)
            dataset_df: Original DataFrame for metadata
            format: Format to save ("csv", "json", "both")
            
        Returns:
            Dictionary with file paths
        """
        saved_files = {}
        
        # Create dataset hash if DataFrame provided
        dataset_hash = ""
        if dataset_df is not None:
            dataset_hash = f"_{self._create_dataset_hash(dataset_df)}"
        
        # Base filename
        base_filename = f"{dataset_name}_predictions{dataset_hash}"
        
        # Prepare prediction data
        prediction_data = {
            "metadata": {
                "experiment_id": self.experiment_id,
                "model_name": result.model_name,
                "model_type": result.model_type.value if hasattr(result.model_type, 'value') else str(result.model_type),
                "classification_type": result.classification_type.value if hasattr(result.classification_type, 'value') else str(result.classification_type),
                "timestamp": datetime.now().isoformat(),
                "dataset_name": dataset_name,
                "dataset_hash": dataset_hash.lstrip("_") if dataset_hash else "",
                "num_predictions": len(result.predictions),
                "git_commit": self.metadata.git_commit
            },
            "predictions": result.predictions,
            "probabilities": result.probabilities if hasattr(result, 'probabilities') else None,
            "confidence_scores": result.confidence_scores if hasattr(result, 'confidence_scores') else None,
            "metrics": result.metadata.get('metrics', {}) if result.metadata else {}
        }
        
        # Save as JSON
        if format in ["json", "both"]:
            json_path = self.predictions_dir / f"{base_filename}.json"
            with open(json_path, 'w') as f:
                json.dump(prediction_data, f, indent=2, default=str)
            saved_files["json"] = str(json_path)
            self.logger.info(f"Saved predictions to JSON: {json_path}")
        
        # Save as CSV
        if format in ["csv", "both"]:
            csv_path = self.predictions_dir / f"{base_filename}.csv"
            
            # Convert to DataFrame
            if dataset_df is not None:
                # Include original data with predictions
                df_with_predictions = dataset_df.copy()
                df_with_predictions['predicted'] = result.predictions
                
                if result.probabilities:
                    df_with_predictions['probabilities'] = [
                        json.dumps(prob) if isinstance(prob, dict) else str(prob) 
                        for prob in result.probabilities
                    ]
                
                if hasattr(result, 'confidence_scores') and result.confidence_scores:
                    df_with_predictions['confidence'] = result.confidence_scores
                
                df_with_predictions.to_csv(csv_path, index=False)
            else:
                # Just predictions
                pred_df = pd.DataFrame({
                    'predictions': result.predictions
                })
                if result.probabilities:
                    pred_df['probabilities'] = [
                        json.dumps(prob) if isinstance(prob, dict) else str(prob) 
                        for prob in result.probabilities
                    ]
                pred_df.to_csv(csv_path, index=False)
            
            saved_files["csv"] = str(csv_path)
            self.logger.info(f"Saved predictions to CSV: {csv_path}")
        
        return saved_files
    
    def save_metrics(self, 
                    metrics: Dict[str, Any],
                    dataset_name: str,
                    model_name: str = "") -> str:
        """Save evaluation metrics.
        
        Args:
            metrics: Metrics dictionary
            dataset_name: Name of the dataset
            model_name: Name of the model
            
        Returns:
            Path to saved metrics file
        """
        # Create metrics data
        metrics_data = {
            "metadata": {
                "experiment_id": self.experiment_id,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "timestamp": datetime.now().isoformat(),
                "git_commit": self.metadata.git_commit
            },
            "metrics": metrics
        }
        
        # Save as YAML (human-readable)
        filename = f"{dataset_name}_metrics"
        if model_name:
            filename = f"{model_name}_{filename}"
        
        metrics_path = self.metrics_dir / f"{filename}.yaml"
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics_data, f, default_flow_style=False)
        
        self.logger.info(f"Saved metrics to: {metrics_path}")
        return str(metrics_path)
    
    def save_model_config(self, config: Dict[str, Any], model_name: str = "") -> str:
        """Save model configuration.
        
        Args:
            config: Model configuration
            model_name: Name of the model
            
        Returns:
            Path to saved config file
        """
        filename = f"{model_name}_config.yaml" if model_name else "model_config.yaml"
        config_path = self.experiment_dir / filename
        
        # Add metadata
        config_with_metadata = {
            "metadata": {
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name
            },
            "config": config
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_with_metadata, f, default_flow_style=False)
        
        self.logger.info(f"Saved model config to: {config_path}")
        return str(config_path)
    
    def save_model_artifacts(self, 
                           model_object: Any,
                           model_name: str,
                           format: str = "pickle") -> str:
        """Save model artifacts.
        
        Args:
            model_object: Model object to save
            model_name: Name of the model
            format: Format to save ("pickle", "torch", "joblib")
            
        Returns:
            Path to saved model
        """
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        if format == "pickle":
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_object, f)
        
        elif format == "torch":
            import torch
            model_path = model_dir / "model.pth"
            torch.save(model_object, model_path)
        
        elif format == "joblib":
            import joblib
            model_path = model_dir / "model.joblib"
            joblib.dump(model_object, model_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved model artifacts to: {model_path}")
        return str(model_path)
    
    def save_experiment_summary(self, 
                              training_results: Dict[str, Any],
                              test_results: Optional[Dict[str, Any]] = None) -> str:
        """Save comprehensive experiment summary.
        
        Args:
            training_results: Training results
            test_results: Optional test results
            
        Returns:
            Path to summary file
        """
        summary = {
            "experiment_metadata": asdict(self.metadata),
            "training_results": training_results,
            "test_results": test_results,
            "file_structure": self._get_experiment_files(),
            "created_at": datetime.now().isoformat()
        }
        
        summary_path = self.experiment_dir / "experiment_summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        self.logger.info(f"Saved experiment summary to: {summary_path}")
        return str(summary_path)
    
    def _get_experiment_files(self) -> Dict[str, List[str]]:
        """Get list of all files created in the experiment."""
        files = {}
        
        for dir_name, dir_path in [
            ("predictions", self.predictions_dir),
            ("models", self.models_dir),
            ("metrics", self.metrics_dir),
            ("logs", self.logs_dir),
            ("plots", self.plots_dir)
        ]:
            if dir_path.exists():
                files[dir_name] = [
                    str(f.relative_to(self.experiment_dir)) 
                    for f in dir_path.rglob("*") if f.is_file()
                ]
            else:
                files[dir_name] = []
        
        return files
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """Get information about the current experiment."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_dir": str(self.experiment_dir),
            "created_at": self.metadata.timestamp,
            "directories": {
                "predictions": str(self.predictions_dir),
                "models": str(self.models_dir),
                "metrics": str(self.metrics_dir),
                "logs": str(self.logs_dir),
                "plots": str(self.plots_dir),
                "cache": str(self.cache_dir)
            }
        }
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments in the output directory."""
        experiments = []
        experiments_dir = self.base_output_dir / "experiments"
        
        if experiments_dir.exists():
            for exp_dir in experiments_dir.iterdir():
                if exp_dir.is_dir():
                    summary_file = exp_dir / "experiment_summary.yaml"
                    if summary_file.exists():
                        try:
                            with open(summary_file, 'r') as f:
                                summary = yaml.safe_load(f)
                            experiments.append({
                                "experiment_id": exp_dir.name,
                                "path": str(exp_dir),
                                "metadata": summary.get("experiment_metadata", {}),
                                "created_at": summary.get("created_at", "")
                            })
                        except Exception:
                            experiments.append({
                                "experiment_id": exp_dir.name,
                                "path": str(exp_dir),
                                "metadata": {},
                                "created_at": ""
                            })
        
        return sorted(experiments, key=lambda x: x.get("created_at", ""), reverse=True)


class ModelResultsManager:
    """Helper class for model-specific results management."""
    
    def __init__(self, results_manager: ResultsManager, model_name: str):
        """Initialize model-specific results manager.
        
        Args:
            results_manager: Main results manager
            model_name: Name of the model
        """
        self.results_manager = results_manager
        self.model_name = model_name
    
    def save_training_results(self, 
                            train_result: ClassificationResult,
                            val_result: Optional[ClassificationResult] = None,
                            train_df: Optional[pd.DataFrame] = None,
                            val_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Save training results for a model.
        
        Args:
            train_result: Training predictions
            val_result: Validation predictions
            train_df: Training DataFrame
            val_df: Validation DataFrame
            
        Returns:
            Dictionary with saved file paths
        """
        saved_files = {}
        
        # Save training predictions
        train_files = self.results_manager.save_predictions(
            train_result, "train", train_df
        )
        saved_files.update({f"train_{k}": v for k, v in train_files.items()})
        
        # Save validation predictions
        if val_result:
            val_files = self.results_manager.save_predictions(
                val_result, "val", val_df
            )
            saved_files.update({f"val_{k}": v for k, v in val_files.items()})
        
        # Save metrics
        if hasattr(train_result, 'metadata') and train_result.metadata:
            train_metrics = train_result.metadata.get('metrics', {})
            if train_metrics:
                metrics_file = self.results_manager.save_metrics(
                    train_metrics, "train", self.model_name
                )
                saved_files["train_metrics"] = metrics_file
        
        if val_result and hasattr(val_result, 'metadata') and val_result.metadata:
            val_metrics = val_result.metadata.get('metrics', {})
            if val_metrics:
                metrics_file = self.results_manager.save_metrics(
                    val_metrics, "val", self.model_name
                )
                saved_files["val_metrics"] = metrics_file
        
        return saved_files
    
    def save_test_results(self, 
                         test_result: ClassificationResult,
                         test_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Save test results for a model.
        
        Args:
            test_result: Test predictions
            test_df: Test DataFrame
            
        Returns:
            Dictionary with saved file paths
        """
        saved_files = {}
        
        # Save test predictions
        test_files = self.results_manager.save_predictions(
            test_result, "test", test_df
        )
        saved_files.update({f"test_{k}": v for k, v in test_files.items()})
        
        # Save test metrics
        if hasattr(test_result, 'metadata') and test_result.metadata:
            test_metrics = test_result.metadata.get('metrics', {})
            if test_metrics:
                metrics_file = self.results_manager.save_metrics(
                    test_metrics, "test", self.model_name
                )
                saved_files["test_metrics"] = metrics_file
        
        return saved_files