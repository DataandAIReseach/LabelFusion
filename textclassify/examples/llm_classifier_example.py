"""Example demonstrating how to use the LLM-based classifiers."""

import asyncio
import pandas as pd
from typing import Dict, Any
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.core.types import ClassificationResult

# Example configuration
def create_config(is_multi_label: bool = False) -> Dict[str, Any]:
    return {
        'model_name': 'gpt-4',
        'parameters': {
            'batch_size': 5,  # Process 5 texts at a time
            'threshold': 0.5,  # Confidence threshold for multi-label
        },
        'label_type': 'multiple' if is_multi_label else 'single'
    }

# Example training data for sentiment analysis (single-label)
def get_sentiment_data() -> pd.DataFrame:
    return pd.DataFrame({
        'text': [
            "This product exceeded my expectations, absolutely fantastic!",
            "Not worth the money, very disappointed with the quality.",
            "It's okay, nothing special but gets the job done.",
        ],
        'label': ['positive', 'negative', 'neutral']
    })

# Example training data for topic classification (multi-label)
def get_topic_data() -> pd.DataFrame:
    return pd.DataFrame({
        'text': [
            "The new AI model shows impressive performance on both NLP and computer vision tasks.",
            "The company's stock price dropped following poor quarterly financial results.",
            "Scientists discover new planet that could potentially support life.",
        ],
        'labels': [
            ['technology', 'artificial_intelligence'],
            ['business', 'finance'],
            ['science', 'astronomy']
        ]
    })

async def run_single_label_example():
    """Example of single-label classification (sentiment analysis)."""
    print("\n=== Single-Label Classification Example (Sentiment) ===")
    
    # Create classifier
    config = create_config(is_multi_label=False)
    classifier = OpenAIClassifier(config)
    
    # Prepare data
    train_df = get_sentiment_data()
    
    # Texts to classify
    texts = [
        "I'm really impressed with the build quality and features!",
        "The service was terrible and the staff was rude.",
    ]
    
    # Get predictions
    result: ClassificationResult = await classifier.predict_async(
        texts=texts,
        train_df=train_df,
        text_column='text',
        label_column='label',
        few_shot_mode='few_shot'  # Use few-shot learning
    )
    
    # Print results
    print("\nPredictions:")
    for text, prediction in zip(texts, result.predictions):
        print(f"\nText: {text}")
        print(f"Predicted sentiment: {prediction}")

async def run_multi_label_example():
    """Example of multi-label classification (topic classification)."""
    print("\n=== Multi-Label Classification Example (Topics) ===")
    
    # Create classifier
    config = create_config(is_multi_label=True)
    classifier = OpenAIClassifier(config)
    
    # Prepare data
    train_df = get_topic_data()
    
    # Texts to classify
    texts = [
        "The startup leverages machine learning to optimize renewable energy distribution.",
        "New study reveals impact of diet and exercise on longevity.",
    ]
    
    # Get predictions
    result: ClassificationResult = await classifier.predict_async(
        texts=texts,
        train_df=train_df,
        text_column='text',
        label_column='labels',
        few_shot_mode='few_shot'  # Use few-shot learning
    )
    
    # Print results
    print("\nPredictions:")
    for text, prediction in zip(texts, result.predictions):
        print(f"\nText: {text}")
        print(f"Predicted topics: {', '.join(prediction)}")

async def main():
    """Run both examples."""
    # Run single-label example
    await run_single_label_example()
    
    # Run multi-label example
    await run_multi_label_example()

if __name__ == "__main__":
    asyncio.run(main())
