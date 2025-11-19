"""Custom exceptions for the textclassify package."""
from typing import Optional


class TextClassifyError(Exception):
    """Base exception for all textclassify errors."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class ModelNotFoundError(TextClassifyError):
    """Raised when a requested model is not found or not available."""
    
    def __init__(self, model_name: str):
        message = f"Model '{model_name}' not found or not available"
        super().__init__(message, "MODEL_NOT_FOUND")
        self.model_name = model_name


class ConfigurationError(TextClassifyError):
    """Raised when there's an error in configuration."""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_key = config_key


class APIError(TextClassifyError):
    """Raised when there's an error with external API calls."""
    
    def __init__(self, message: str, provider: str = None, status_code: int = None):
        super().__init__(message, "API_ERROR")
        self.provider = provider
        self.status_code = status_code


class ModelTrainingError(TextClassifyError):
    """Raised when there's an error during model training."""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(message, "MODEL_TRAINING_ERROR")
        self.model_name = model_name


class PredictionError(TextClassifyError):
    """Raised when there's an error during prediction."""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(message, "PREDICTION_ERROR")
        self.model_name = model_name


class ValidationError(TextClassifyError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field_name: str = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field_name = field_name


class EnsembleError(TextClassifyError):
    """Raised when there's an error in ensemble operations."""
    
    def __init__(self, message: str, ensemble_method: str = None):
        super().__init__(message, "ENSEMBLE_ERROR")
        self.ensemble_method = ensemble_method


class PersistenceError(TextClassifyError):
    """Raised when there's an error in persistence operations (caching, saving, loading)."""
    
    def __init__(self, message: str, operation: str = None):
        super().__init__(message, "PERSISTENCE_ERROR")
        self.operation = operation


class CudaOutOfMemoryError(ModelTrainingError):
    """Raised when CUDA/GPU memory allocation fails during model operations.

    This exception wraps PyTorch's OutOfMemoryError and provides structured
    information that callers can use to present clearer diagnostics and
    remediation steps to users.
    """

    def __init__(
        self,
        message: str,
        attempted_allocation: Optional[str] = None,
        total_memory: Optional[str] = None,
        free_memory: Optional[str] = None,
        process_id: Optional[int] = None,
        suggestion: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        full_message = message
        details = []
        if attempted_allocation:
            details.append(f"attempted_allocation={attempted_allocation}")
        if total_memory:
            details.append(f"total_memory={total_memory}")
        if free_memory:
            details.append(f"free_memory={free_memory}")
        if process_id is not None:
            details.append(f"process_id={process_id}")
        if details:
            full_message = full_message + " | " + ", ".join(details)

        super().__init__(full_message, model_name)

        self.attempted_allocation = attempted_allocation
        self.total_memory = total_memory
        self.free_memory = free_memory
        self.process_id = process_id
        self.suggestion = suggestion or (
            "Possible remedies: free GPU memory, set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, "
            "reduce batch size, or run on CPU."
        )
        self.model_name = model_name

