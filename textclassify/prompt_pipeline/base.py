from abc import ABC, abstractmethod
from .prompt_warehouse import PromptWarehouse


class PromptPipeline(ABC):
    """Abstract base class for prompt pipelines."""

    @abstractmethod
    def get_warehouse(self, text: str) -> PromptWarehouse:
        """
        Return a ready-to-use PromptWarehouse for the given text.

        Concrete implementations decide how language detection
        and translation are handled.

        Args:
            text: Input text to base the warehouse on

        Returns:
            PromptWarehouse with prompts in the appropriate language
        """
        pass