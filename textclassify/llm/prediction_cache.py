"""Prediction cache and persistence manager for LLM classifiers."""

import os
import csv
import json
import pickle
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pandas as pd

from ..core.exceptions import PersistenceError


class LLMPredictionCache:
    """Manages caching and persistence of LLM predictions to enable recovery from failures."""
    
    def __init__(
        self,
        cache_dir: str = "cache/llm",
        session_id: Optional[str] = None,
        auto_save_interval: int = 5,  # Save every N predictions
        enable_compression: bool = True,
        verbose: bool = True
    ):
        """Initialize the prediction cache.
        
        Args:
            cache_dir: Directory to store cache files
            session_id: Unique identifier for this prediction session
            auto_save_interval: Number of predictions before auto-saving
            enable_compression: Whether to compress cache files
            verbose: Whether to log cache operations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = session_id or self._generate_session_id()
        self.auto_save_interval = auto_save_interval
        self.enable_compression = enable_compression
        self.verbose = verbose
        
        # Cache data structures
        self.predictions_cache: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "last_updated": None
        }
        
        # File paths
        self.predictions_file = self.cache_dir / f"predictions_{self.session_id}.csv"
        self.metadata_file = self.cache_dir / f"metadata_{self.session_id}.json"
        self.cache_file = self.cache_dir / f"cache_{self.session_id}.pkl"
        
        # Counters
        self._prediction_count = 0
        self._last_save_count = 0
        
        # Setup logging
        if self.verbose:
            self.logger = logging.getLogger(f"LLMPredictionCache_{self.session_id}")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        
        # Try to load existing cache
        self._load_existing_cache()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID based on timestamp and random component."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_hash = hashlib.md5(str(datetime.now().microsecond).encode()).hexdigest()[:8]
        return f"{timestamp}_{random_hash}"
    
    def _load_existing_cache(self) -> None:
        """Load existing cache data if available."""
        try:
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    saved_metadata = json.load(f)
                    self.metadata.update(saved_metadata)
                    if self.verbose:
                        self.logger.info(f"Loaded existing metadata: {len(saved_metadata)} items")
            else:
                # If there's no metadata for this session id, try to find the most
                # recent metadata/cache files in the cache directory and load them.
                meta_files = sorted(self.cache_dir.glob('metadata_*.json'), reverse=True)
                cache_files = sorted(self.cache_dir.glob('cache_*.pkl'), reverse=True)
                if meta_files or cache_files:
                    # Prefer metadata file to pick session id, otherwise pick cache file
                    chosen_meta = meta_files[0] if meta_files else None
                    chosen_cache = cache_files[0] if cache_files else None
                    if chosen_meta:
                        try:
                            with open(chosen_meta, 'r') as f:
                                saved_metadata = json.load(f)
                                self.metadata.update(saved_metadata)
                                # update session id and file paths to match discovered file
                                sid = chosen_meta.stem.replace('metadata_', '')
                                self.session_id = sid
                                self.metadata_file = chosen_meta
                                self.predictions_file = self.cache_dir / f"predictions_{sid}.csv"
                                self.cache_file = self.cache_dir / f"cache_{sid}.pkl"
                                if self.verbose:
                                    self.logger.info(f"Auto-loaded metadata from {chosen_meta.name}")
                        except Exception:
                            pass
                    elif chosen_cache:
                        # derive session id from cache filename
                        sid = chosen_cache.stem.replace('cache_', '')
                        self.session_id = sid
                        self.metadata_file = self.cache_dir / f"metadata_{sid}.json"
                        self.predictions_file = self.cache_dir / f"predictions_{sid}.csv"
                        self.cache_file = chosen_cache
                        if self.verbose:
                            self.logger.info(f"Auto-selected cache file {chosen_cache.name}")
            
            # Load cache
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.predictions_cache = pickle.load(f)
                    if self.verbose:
                        self.logger.info(f"Loaded existing cache: {len(self.predictions_cache)} predictions")
            
            # Update counters
            self._prediction_count = len(self.predictions_cache)
            self._last_save_count = self._prediction_count
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Could not load existing cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for the input text to use as cache key."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def has_prediction(self, text: str) -> bool:
        """Check if we already have a prediction for this text."""
        text_hash = self._get_text_hash(text)
        return text_hash in self.predictions_cache
    
    def get_prediction(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction for text if available."""
        text_hash = self._get_text_hash(text)
        return self.predictions_cache.get(text_hash)
    
    def store_prediction(
        self,
        text: str,
        prediction: List[int],
        response_text: str,
        prompt: str,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a prediction result in the cache.
        
        Args:
            text: Original input text
            prediction: Binary vector prediction
            response_text: Raw LLM response
            prompt: The prompt sent to LLM
            success: Whether the prediction was successful
            error_message: Error message if prediction failed
            metadata: Additional metadata to store
        """
        text_hash = self._get_text_hash(text)
        
        prediction_data = {
            "text_hash": text_hash,
            "text": text,
            "prediction": prediction,
            "response_text": response_text,
            "prompt": prompt,
            "success": success,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        # If an identifier was provided inside metadata (common keys: id, index, row_id),
        # lift it to a top-level 'id' field for easier consumption by CSV/JSON exports.
        id_value = None
        if metadata:
            for key in ("id", "index", "row_id", "uid"):
                if key in metadata:
                    id_value = metadata.get(key)
                    break
        if id_value is not None:
            prediction_data["id"] = id_value
        
        # Store in cache
        self.predictions_cache[text_hash] = prediction_data
        self._prediction_count += 1
        
        # Update metadata
        self.metadata["total_predictions"] = self._prediction_count
        if success:
            self.metadata["successful_predictions"] += 1
        else:
            self.metadata["failed_predictions"] += 1
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        # Auto-save if needed
        if self._prediction_count - self._last_save_count >= self.auto_save_interval:
            self.save_cache()
        
        if self.verbose:
            status = "✓" if success else "✗"
            self.logger.info(f"{status} Cached prediction {self._prediction_count} (hash: {text_hash[:8]})")
    
    def save_cache(self) -> None:
        """Save current cache to disk."""
        try:
            # Save CSV file for human readability
            self._save_predictions_csv()
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save pickle cache for fast loading
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.predictions_cache, f)
            
            self._last_save_count = self._prediction_count
            
            if self.verbose:
                self.logger.info(f"Saved cache: {len(self.predictions_cache)} predictions to {self.cache_dir}")
                
        except Exception as e:
            raise PersistenceError(f"Failed to save cache: {e}")
    
    def _save_predictions_csv(self) -> None:
        """Save predictions to CSV file for easy inspection."""
        if not self.predictions_cache:
            return
        
        # Prepare data for CSV
        csv_data = []
        for text_hash, pred_data in self.predictions_cache.items():
            row = {
                # include id if available to make it easy to map back to original rows
                "id": pred_data.get("id", ""),
                "text_hash": text_hash,
                "text": pred_data["text"][:500] + "..." if len(pred_data["text"]) > 500 else pred_data["text"],  # Truncate for CSV
                "prediction": json.dumps(pred_data["prediction"]),
                "response_text": pred_data["response_text"],
                "success": pred_data["success"],
                "error_message": pred_data.get("error_message", ""),
                "timestamp": pred_data["timestamp"]
            }
            csv_data.append(row)
        
        # Write CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(self.predictions_file, index=False)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the current cache."""
        return {
            "session_id": self.session_id,
            "total_predictions": len(self.predictions_cache),
            "successful_predictions": sum(1 for p in self.predictions_cache.values() if p["success"]),
            "failed_predictions": sum(1 for p in self.predictions_cache.values() if not p["success"]),
            "cache_size_mb": self._get_cache_size_mb(),
            "cache_directory": str(self.cache_dir),
            "last_updated": self.metadata.get("last_updated", "Never")
        }
    
    def _get_cache_size_mb(self) -> float:
        """Calculate total cache size in MB."""
        total_size = 0
        for file_path in [self.predictions_file, self.metadata_file, self.cache_file]:
            if file_path.exists():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)
    
    def export_predictions(self, output_file: str, format: str = "csv") -> None:
        """Export predictions to a file.
        
        Args:
            output_file: Path to output file
            format: Export format ('csv', 'json', 'xlsx')
        """
        if not self.predictions_cache:
            raise ValueError("No predictions to export")
        
        output_path = Path(output_file)
        
        # Prepare data
        export_data = []
        for pred_data in self.predictions_cache.values():
            row = {
                # include id if present
                "id": pred_data.get("id"),
                "text": pred_data["text"],
                "prediction": pred_data["prediction"],
                "response_text": pred_data["response_text"],
                "success": pred_data["success"],
                "timestamp": pred_data["timestamp"]
            }
            if pred_data.get("error_message"):
                row["error_message"] = pred_data["error_message"]
            export_data.append(row)
        
        if format.lower() == "csv":
            df = pd.DataFrame(export_data)
            df.to_csv(output_path, index=False)
        elif format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == "xlsx":
            df = pd.DataFrame(export_data)
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        if self.verbose:
            self.logger.info(f"Exported {len(export_data)} predictions to {output_path}")
    
    def clear_cache(self, confirm: bool = False) -> None:
        """Clear all cached predictions.
        
        Args:
            confirm: Must be True to actually clear the cache
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear cache")
        
        self.predictions_cache.clear()
        self._prediction_count = 0
        self._last_save_count = 0
        
        # Remove files
        for file_path in [self.predictions_file, self.metadata_file, self.cache_file]:
            if file_path.exists():
                file_path.unlink()
        
        if self.verbose:
            self.logger.info("Cache cleared")
    
    def get_failed_predictions(self) -> List[Dict[str, Any]]:
        """Get all failed predictions for retry."""
        return [pred for pred in self.predictions_cache.values() if not pred["success"]]
    
    def get_successful_predictions(self) -> List[Tuple[str, List[int]]]:
        """Get all successful predictions as (text, prediction) tuples."""
        return [
            (pred["text"], pred["prediction"])
            for pred in self.predictions_cache.values()
            if pred["success"]
        ]
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-save cache."""
        self.save_cache()
        if self.verbose:
            self.logger.info(f"Session {self.session_id} completed. Final stats: {self.get_cache_stats()}")