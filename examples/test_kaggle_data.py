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
    df = pd.read_csv('data/ecommerceDataset.csv', encoding='latin1')
    
    # Drop first column and reorder/rename columns
    cols = df.columns.tolist()
    df = df[[cols[1], cols[0]]]  # Swap second and third columns
    df.columns = ['text', 'label']  # Rename columns
    
    # Convert label column to dummy variables and add them to the DataFrame
    label_dummies = pd.get_dummies(df['label'], prefix='')
    df = pd.concat([df[['text']], label_dummies], axis=1)  # Keep only 'Text' column and add dummies

    # Get unique labels
    label_columns = label_dummies.columns.tolist()
    print(f"Available labels: {label_columns}")
    
    # Create configuration
    config = Config()
    config.model_type = ModelType.LLM
    config.parameters = {
        'model': 'o4-mini',
        'temperature': 1,
        'max_tokens': 150,
        'batch_size': 5
    }
    config.api_key = os.getenv('OPENAI_API_KEY')
    
    # Initialize classifier with column specifications
    classifier = OpenAIClassifier(
        config=config,
        text_column='text',
        label_columns=label_columns  # Use the dummy column names as labels
    )
    
    # Take first 10 rows for training and next 5 for testing
    train_df = df.iloc[:10]
    test_df = df.iloc[10:15]
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    try:
        # Make predictions
        result = classifier.predict(
            train_df=train_df,
            test_df=test_df
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