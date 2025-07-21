"""Simple test for OpenAI text classification."""

import os
import pandas as pd
from dotenv import load_dotenv
import asyncio
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.config.settings import Config
from textclassify.core.types import ModelType

# Load environment variables
load_dotenv()

def main():
    # Load and prepare data
    df = pd.read_csv('data/alldata_1_for_kaggle.csv', encoding='latin1')
    
    # Create configuration
    config = Config()
    config.model_type = ModelType.LLM
    config.parameters = {
        'model': 'gpt-3.5-turbo',
        'temperature': 0.1,
        'max_tokens': 150,
        'batch_size': 5
    }
    config.api_key = os.getenv('OPENAI_API_KEY')
    
    # Initialize classifier with column specifications
    classifier = OpenAIClassifier(
        config=config,
        text_column='text',  # Adjust this to match your actual text column name
        label_columns=['label']  # Adjust this to match your actual label column names
    )
    
    # Take first 10 rows for training and next 5 for testing
    train_df = df.iloc[:10]
    test_df = df.iloc[10:15]
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    try:
        # Make predictions
        result = classifier.predict(
            df=test_df,
            train_df=train_df,
            text_column='text'
        )
        
        # Print results
        print("\nResults:")
        print("=" * 80)
        for idx, (text, pred) in enumerate(zip(test_df['text'], result.predictions)):
            print(f"\nTest {idx + 1}:")
            print(f"Text: {text[:100]}...")
            print(f"Prediction: {pred}")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()