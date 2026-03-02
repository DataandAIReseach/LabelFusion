"""Prompt templates for LLM-based text classification."""

from typing import List, Dict, Any
from ..core.types import ClassificationType


class PromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, classification_type: ClassificationType):
        self.classification_type = classification_type
    
    def format_prompt(self, text: str, classes: List[str], examples: List[Dict[str, Any]] = None) -> str:
        """Format the prompt for classification.
        
        Args:
            text: Text to classify
            classes: List of possible classes
            examples: Optional few-shot examples
            
        Returns:
            Formatted prompt string
        """
        raise NotImplementedError


class MultiClassPromptTemplate(PromptTemplate):
    """Prompt template for multi-class classification."""
    
    def __init__(self):
        super().__init__(ClassificationType.MULTI_CLASS)
    
    def format_prompt(self, text: str, classes: List[str], examples: List[Dict[str, Any]] = None) -> str:
        """Format prompt for multi-class classification."""
        
        # Base instruction
        prompt = f"""You are a text classification expert. Your task is to classify the given text into exactly ONE of the following categories:

Categories:
{self._format_classes(classes)}

Instructions:
1. Read the text carefully
2. Choose the SINGLE most appropriate category
3. Respond with ONLY the category name, nothing else
4. If uncertain, choose the closest match

"""
        
        # Add examples if provided
        if examples:
            prompt += "Examples:\n"
            for example in examples:
                prompt += f"Text: {example['text']}\nCategory: {example['label']}\n\n"
        
        # Add the text to classify
        prompt += f"Text to classify: {text}\nCategory:"
        
        return prompt
    
    def _format_classes(self, classes: List[str]) -> str:
        """Format the list of classes."""
        return "\n".join(f"- {cls}" for cls in classes)


class MultiLabelPromptTemplate(PromptTemplate):
    """Prompt template for multi-label classification."""
    
    def __init__(self):
        super().__init__(ClassificationType.MULTI_LABEL)
    
    def format_prompt(self, text: str, classes: List[str], examples: List[Dict[str, Any]] = None) -> str:
        """Format prompt for multi-label classification."""
        
        # Base instruction
        prompt = f"""You are a text classification expert. Your task is to classify the given text into one or more of the following categories:

Categories:
{self._format_classes(classes)}

Instructions:
1. Read the text carefully
2. Select ALL categories that apply to the text
3. You can select multiple categories if they all apply
4. If no categories apply, respond with "NONE"
5. Respond with category names separated by commas, nothing else

"""
        
        # Add examples if provided
        if examples:
            prompt += "Examples:\n"
            for example in examples:
                labels = ", ".join(example['label']) if isinstance(example['label'], list) else example['label']
                prompt += f"Text: {example['text']}\nCategories: {labels}\n\n"
        
        # Add the text to classify
        prompt += f"Text to classify: {text}\nCategories:"
        
        return prompt
    
    def _format_classes(self, classes: List[str]) -> str:
        """Format the list of classes."""
        return "\n".join(f"- {cls}" for cls in classes)


class ProbabilityPromptTemplate(PromptTemplate):
    """Prompt template for getting classification probabilities."""
    
    def __init__(self, classification_type: ClassificationType):
        super().__init__(classification_type)
    
    def format_prompt(self, text: str, classes: List[str], examples: List[Dict[str, Any]] = None) -> str:
        """Format prompt for probability-based classification."""
        
        if self.classification_type == ClassificationType.MULTI_CLASS:
            return self._format_multiclass_probability_prompt(text, classes, examples)
        else:
            return self._format_multilabel_probability_prompt(text, classes, examples)
    
    def _format_multiclass_probability_prompt(self, text: str, classes: List[str], examples: List[Dict[str, Any]] = None) -> str:
        """Format prompt for multi-class probability classification."""
        
        prompt = f"""You are a text classification expert. Your task is to classify the given text and provide confidence scores for each category.

Categories:
{self._format_classes(classes)}

Instructions:
1. Read the text carefully
2. Assign a confidence score (0.0 to 1.0) for each category
3. Scores should sum to 1.0
4. Respond in JSON format: {{"category1": score1, "category2": score2, ...}}
5. Use exactly the category names provided

"""
        
        # Add examples if provided
        if examples:
            prompt += "Examples:\n"
            for example in examples:
                prompt += f"Text: {example['text']}\nScores: {example.get('scores', 'N/A')}\n\n"
        
        # Add the text to classify
        prompt += f"Text to classify: {text}\nScores:"
        
        return prompt
    
    def _format_multilabel_probability_prompt(self, text: str, classes: List[str], examples: List[Dict[str, Any]] = None) -> str:
        """Format prompt for multi-label probability classification."""
        
        prompt = f"""You are a text classification expert. Your task is to classify the given text and provide confidence scores for each category.

Categories:
{self._format_classes(classes)}

Instructions:
1. Read the text carefully
2. Assign a confidence score (0.0 to 1.0) for each category independently
3. Each score represents how likely the category applies to the text
4. Scores do NOT need to sum to 1.0 (this is multi-label classification)
5. Respond in JSON format: {{"category1": score1, "category2": score2, ...}}
6. Use exactly the category names provided

"""
        
        # Add examples if provided
        if examples:
            prompt += "Examples:\n"
            for example in examples:
                prompt += f"Text: {example['text']}\nScores: {example.get('scores', 'N/A')}\n\n"
        
        # Add the text to classify
        prompt += f"Text to classify: {text}\nScores:"
        
        return prompt
    
    def _format_classes(self, classes: List[str]) -> str:
        """Format the list of classes."""
        return "\n".join(f"- {cls}" for cls in classes)


def get_prompt_template(classification_type: ClassificationType, with_probabilities: bool = False) -> PromptTemplate:
    """Get the appropriate prompt template.
    
    Args:
        classification_type: Type of classification (multi-class or multi-label)
        with_probabilities: Whether to request probability scores
        
    Returns:
        Appropriate prompt template instance
    """
    if with_probabilities:
        return ProbabilityPromptTemplate(classification_type)
    elif classification_type == ClassificationType.MULTI_CLASS:
        return MultiClassPromptTemplate()
    else:
        return MultiLabelPromptTemplate()

