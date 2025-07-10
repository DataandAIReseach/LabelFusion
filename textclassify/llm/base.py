"""Base class for LLM-based text classifiers."""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from ..core.base import AsyncBaseClassifier
from ..core.types import ClassificationResult, ClassificationType, ModelType, TrainingData
from ..core.exceptions import PredictionError, ValidationError, APIError
from .prompts import get_prompt_template
from ..prompt_engineer import PromptEngineer

class BaseLLMClassifier(AsyncBaseClassifier):
    """Base class for all LLM-based text classifiers."""
    
    def __init__(self, config):

        super().__init__(config)
        self.config.model_type = ModelType.LLM
        self.examples = []
        self.context = None
        self.label_definitions = {}
        self.prompt_engineer = PromptEngineer()

        

    def fit(
        self,
        training_data: TrainingData,
        context: str = None,
        label_definitions: dict = None
    ) -> None:
        """Fit the LLM classifier (stores examples for few-shot learning).

        Args:
            training_data: Training data containing texts and labels
            context: Optional context string for prompt engineering
            label_definitions: Optional dict of label definitions
        """
        if context is not None:
            self.context = context
            self.prompt_engineer.context = context
        if label_definitions is not None:
            self.label_definitions = label_definitions
            self.prompt_engineer.label_definitions = label_definitions

        self.classification_type = training_data.classification_type

        # Extract unique classes
        if self.classification_type == ClassificationType.MULTI_CLASS:
            self.classes_ = list(set(training_data.labels))
            label_type = "single"
        else:
            all_labels = []
            for label_list in training_data.labels:
                all_labels.extend(label_list)
            self.classes_ = list(set(all_labels))
            label_type = "multiple"

        self.prompt_engineer.set_labels(self.classes_, label_type=label_type)

        # Store examples for few-shot learning (limit to avoid token limits)
        max_examples = self.config.parameters.get('max_examples', 5)
        self.examples = []
        for i, (text, label) in enumerate(zip(training_data.texts, training_data.labels)):
            if i >= max_examples:
                break
            self.examples.append({'text': text, 'label': label})

        self.is_trained = True
    
    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        context: str = None,
        label_definitions: dict = None,
        classification_type: str = None
        ) -> None:
            """
            Fit the classifier using a pandas DataFrame directly.

            Args:
                df: The DataFrame containing the data.
                text_column: Name of the column with input texts.
                label_column: Name of the column with labels.
                context: Optional context string for prompt engineering.
                label_definitions: Optional dict of label definitions.
                classification_type: "multi_class" or "multi_label". If None, inferred automatically.
            """
            texts = df[text_column].tolist()
            labels = df[label_column].tolist()

            # Infer classification type if not provided
            if classification_type is None:
                if all(isinstance(label, str) for label in labels):
                    ctype = ClassificationType.MULTI_CLASS
                elif all(isinstance(label, list) for label in labels):
                    ctype = ClassificationType.MULTI_LABEL
                else:
                    raise ValueError("Could not infer classification type from label column.")
            else:
                if classification_type == "multi_class":
                    ctype = ClassificationType.MULTI_CLASS
                elif classification_type == "multi_label":
                    ctype = ClassificationType.MULTI_LABEL
                else:
                    raise ValueError("classification_type must be 'multi_class' or 'multi_label'.")
    
    def predict(self, texts: List[str]) -> ClassificationResult:
        """Predict labels for texts (synchronous wrapper).
        
        Args:
            texts: List of texts to classify
            
        Returns:
            ClassificationResult with predictions
        """
        return asyncio.run(self.predict_async(texts))
    
    def predict_proba(self, texts: List[str]) -> ClassificationResult:
        """Predict probabilities for texts (synchronous wrapper).
        
        Args:
            texts: List of texts to classify
            
        Returns:
            ClassificationResult with predictions and probabilities
        """
        return asyncio.run(self.predict_proba_async(texts))
    
    async def predict_async(self, texts: List[str]) -> ClassificationResult:
        """Asynchronously predict labels for texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            ClassificationResult with predictions
        """
        self.validate_input(texts)
        
        if not self.is_trained:
            raise PredictionError("Model must be trained before prediction", self.config.model_name)
        
        # Get prompt template
        template = get_prompt_template(self.classification_type, with_probabilities=False)
        
        # Process texts in batches
        batch_size = self.config.batch_size
        all_predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_predictions = await self._predict_batch(batch_texts, template)
            all_predictions.extend(batch_predictions)
        
        return self._create_result(predictions=all_predictions)
    
    async def predict_proba_async(self, texts: List[str]) -> ClassificationResult:
        """Asynchronously predict probabilities for texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            ClassificationResult with predictions and probabilities
        """
        self.validate_input(texts)
        
        if not self.is_trained:
            raise PredictionError("Model must be trained before prediction", self.config.model_name)
        
        # Get probability prompt template
        template = get_prompt_template(self.classification_type, with_probabilities=True)
        
        # Process texts in batches
        batch_size = self.config.batch_size
        all_predictions = []
        all_probabilities = []
        all_confidence_scores = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = await self._predict_proba_batch(batch_texts, template)
            
            for pred, prob, conf in batch_results:
                all_predictions.append(pred)
                all_probabilities.append(prob)
                all_confidence_scores.append(conf)
        
        return self._create_result(
            predictions=all_predictions,
            probabilities=all_probabilities,
            confidence_scores=all_confidence_scores
        )
    
    async def _predict_batch(self, texts: List[str], template) -> List[Union[str, List[str]]]:
        """Predict labels for a batch of texts.
        
        Args:
            texts: Batch of texts to classify
            template: Prompt template to use
            
        Returns:
            List of predictions
        """
        tasks = []
        for text in texts:
            prompt = template.format_prompt(text, self.classes_, self.examples)
            tasks.append(self._call_llm(prompt))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        predictions = []
        for response in responses:
            if isinstance(response, Exception):
                raise PredictionError(f"LLM call failed: {str(response)}", self.config.model_name)
            
            prediction = self._parse_prediction_response(response)
            predictions.append(prediction)
        
        return predictions
    
    async def _predict_proba_batch(self, texts: List[str], template) -> List[tuple]:
        """Predict probabilities for a batch of texts.
        
        Args:
            texts: Batch of texts to classify
            template: Prompt template to use
            
        Returns:
            List of (prediction, probabilities, confidence) tuples
        """
        tasks = []
        for text in texts:
            prompt = template.format_prompt(text, self.classes_, self.examples)
            tasks.append(self._call_llm(prompt))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for response in responses:
            if isinstance(response, Exception):
                raise PredictionError(f"LLM call failed: {str(response)}", self.config.model_name)
            
            prediction, probabilities, confidence = self._parse_probability_response(response)
            results.append((prediction, probabilities, confidence))
        
        return results
    
    def _parse_prediction_response(self, response: str) -> Union[str, List[str]]:
        """Parse the LLM response for predictions.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed prediction(s)
        """
        response = response.strip()
        
        if self.classification_type == ClassificationType.MULTI_CLASS:
            # For multi-class, expect a single class name
            # Clean the response and find the best match
            for class_name in self.classes_:
                if class_name.lower() in response.lower():
                    return class_name
            
            # If no exact match, return the first class as fallback
            return self.classes_[0] if self.classes_ else "unknown"
        
        else:
            # For multi-label, expect comma-separated class names
            if response.upper() == "NONE":
                return []
            
            # Split by comma and clean
            predicted_classes = []
            parts = response.split(',')
            
            for part in parts:
                part = part.strip()
                for class_name in self.classes_:
                    if class_name.lower() in part.lower():
                        predicted_classes.append(class_name)
                        break
            
            return predicted_classes
    
    def _parse_probability_response(self, response: str) -> tuple:
        """Parse the LLM response for probabilities.
        
        Args:
            response: Raw LLM response containing JSON
            
        Returns:
            Tuple of (prediction, probabilities_dict, confidence_score)
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            probabilities = json.loads(json_match.group())
            
            # Validate and normalize probabilities
            normalized_probs = {}
            for class_name in self.classes_:
                # Try to find the class in the response (case-insensitive)
                prob = 0.0
                for key, value in probabilities.items():
                    if key.lower() == class_name.lower():
                        prob = float(value)
                        break
                normalized_probs[class_name] = max(0.0, min(1.0, prob))
            
            # Generate prediction based on probabilities
            if self.classification_type == ClassificationType.MULTI_CLASS:
                prediction = max(normalized_probs, key=normalized_probs.get)
                confidence = normalized_probs[prediction]
            else:
                # For multi-label, use threshold (default 0.5)
                threshold = self.config.parameters.get('threshold', 0.5)
                prediction = [cls for cls, prob in normalized_probs.items() if prob >= threshold]
                confidence = max(normalized_probs.values()) if normalized_probs else 0.0
            
            return prediction, normalized_probs, confidence
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback to simple prediction parsing
            prediction = self._parse_prediction_response(response)
            
            # Create uniform probabilities
            if self.classification_type == ClassificationType.MULTI_CLASS:
                prob_value = 1.0 / len(self.classes_) if self.classes_ else 0.0
                probabilities = {cls: prob_value for cls in self.classes_}
                confidence = prob_value
            else:
                probabilities = {cls: 0.5 for cls in self.classes_}
                confidence = 0.5
            
            return prediction, probabilities, confidence
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API with the given prompt.
        
        This method should be implemented by each specific LLM provider.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response
            
        Raises:
            APIError: If the API call fails
        """
        raise NotImplementedError("Subclasses must implement _call_llm method")

