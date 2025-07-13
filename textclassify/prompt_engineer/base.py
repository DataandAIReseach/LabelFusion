class prompt_engineer:
    """
    Base class for prompt engineering.
    This class provides a template for creating prompt engineers.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_prompt_single_label(self, input_data: str) -> str:
        """
        Generate a prompt based on the input data.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def generate_prompt_multiple_labels(self, input_data: str) -> str:
        """
        Generate a prompt for multiple labels based on the input data.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")