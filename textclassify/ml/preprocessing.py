"""Text preprocessing utilities for ML classifiers."""

import re
import string
from typing import List, Optional, Dict, Any


class TextPreprocessor:
    """Text preprocessing utilities for machine learning models."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_numbers: bool = False,
                 remove_extra_whitespace: bool = True,
                 min_length: int = 1,
                 max_length: Optional[int] = None):
        """Initialize text preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            remove_extra_whitespace: Whether to remove extra whitespace
            min_length: Minimum text length (in characters)
            max_length: Maximum text length (in characters)
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_length = min_length
        self.max_length = max_length
        
        # Compile regex patterns for efficiency
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess a single text.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove numbers
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = self.punctuation_pattern.sub(' ', text)
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Apply length constraints
        if len(text) < self.min_length:
            return ""  # Return empty string for texts that are too short
        
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]
    
    def get_config(self) -> Dict[str, Any]:
        """Get preprocessor configuration.
        
        Returns:
            Dictionary containing preprocessor settings
        """
        return {
            'lowercase': self.lowercase,
            'remove_punctuation': self.remove_punctuation,
            'remove_numbers': self.remove_numbers,
            'remove_extra_whitespace': self.remove_extra_whitespace,
            'min_length': self.min_length,
            'max_length': self.max_length
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TextPreprocessor':
        """Create preprocessor from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            TextPreprocessor instance
        """
        return cls(**config)


def clean_text(text: str, 
               remove_urls: bool = True,
               remove_emails: bool = True,
               remove_mentions: bool = True,
               remove_hashtags: bool = False) -> str:
    """Clean text by removing specific patterns.
    
    Args:
        text: Input text
        remove_urls: Whether to remove URLs
        remove_emails: Whether to remove email addresses
        remove_mentions: Whether to remove @mentions
        remove_hashtags: Whether to remove #hashtags
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove URLs
    if remove_urls:
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = url_pattern.sub(' ', text)
        
        # Also remove www.domain.com patterns
        www_pattern = re.compile(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = www_pattern.sub(' ', text)
    
    # Remove email addresses
    if remove_emails:
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        text = email_pattern.sub(' ', text)
    
    # Remove @mentions
    if remove_mentions:
        mention_pattern = re.compile(r'@\w+')
        text = mention_pattern.sub(' ', text)
    
    # Remove #hashtags
    if remove_hashtags:
        hashtag_pattern = re.compile(r'#\w+')
        text = hashtag_pattern.sub(' ', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_text(text: str) -> str:
    """Normalize text for better processing.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Replace common contractions
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Normalize dashes
    text = text.replace('—', '-').replace('–', '-')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

