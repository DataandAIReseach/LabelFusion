"""Test for both traditional ML and LLM text classification."""

import os
import pandas as pd
from dotenv import load_dotenv
import asyncio
from textclassify.llm.deepseek_classifier import DeepSeekClassifier
from textclassify.llm.gemini_classifier import GeminiClassifier  # Gemini alternative
from textclassify.llm.openai_classifier import OpenAIClassifier  # OpenAI alternative
from textclassify.ml.roberta_classifier import RobertaClassifier  # Traditional ML classifier
from textclassify.config.settings import Config
from textclassify.core.types import ModelType, TrainingData, ModelConfig

# Load environment variables
load_dotenv()

def test_ml_classifier(train_df, test_df, label_columns):
    """Test the traditional RoBERTa ML classifier."""
    print("\n" + "="*80)
    print("TESTING TRADITIONAL ML CLASSIFIER (RoBERTa)")
    print("="*80)
    
    # Create ML configuration
    config = Config()
    config.model_type = ModelType.ML
    config.parameters = {
        'model_name': 'roberta-base',
        'max_length': 512,
        'learning_rate': 2e-5,
        'epochs': 3,
        'batch_size': 8,
        'warmup_steps': 100
    }
    
    # Create training data for ML classifier
    training_data = TrainingData(
        texts=train_df['text'].tolist(),
        labels=train_df[label_columns].values.tolist(),
        label_names=label_columns
    )
    
    # Create model configuration
    model_config = ModelConfig(
        model_name='roberta-base',
        num_labels=len(label_columns),
        max_length=512
    )
    
    # Initialize ML classifier
    ml_classifier = RobertaClassifier(
        config=config,
        text_column='text',
        label_columns=label_columns,
        multi_label=True  # Assuming multi-label classification
    )
    
    try:
        # Train the classifier
        print("Training RoBERTa classifier...")
        ml_classifier.train(training_data, model_config)
        
        # Make predictions
        print("Making predictions...")
        result = ml_classifier.predict(
            train_df=train_df,
            test_df=test_df
        )
        
        # Print results
        print("\nML Classifier Results:")
        print("-" * 60)
        for idx, (text, pred) in enumerate(zip(test_df['text'], result.predictions)):
            print(f"\nTest {idx + 1}:")
            print(f"Text: {text[:100]}...")
            print(f"Prediction: {pred}")
            
        return result
        
    except Exception as e:
        print(f"Error during ML classification: {str(e)}")
        return None

def test_llm_classifier(train_df, test_df, label_columns):
    """Test the LLM classifier (DeepSeek)."""
    print("\n" + "="*80)
    print("TESTING LLM CLASSIFIER (DeepSeek)")
    print("="*80)
    
    # Create LLM configuration
    config = Config()
    config.model_type = ModelType.LLM
    config.parameters = {
        'model': 'deepseek-chat',  # DeepSeek model
        'temperature': 1,
        'max_tokens': 150,
        'batch_size': 5,
    }
    
    # Initialize LLM classifier
    classifier = DeepSeekClassifier(
        config=config,
        text_column='text',
        label_columns=label_columns  # Use the dummy column names as labels
    )
    
    try:
        # Make predictions
        print("Making LLM predictions...")
        result = classifier.predict(
            train_df=train_df,
            test_df=test_df
        )
        
        # Print results
        print("\nLLM Classifier Results:")
        print("-" * 60)
        for idx, (text, pred) in enumerate(zip(test_df['text'], result.predictions)):
            print(f"\nTest {idx + 1}:")
            print(f"Text: {text[:100]}...")
            print(f"Prediction: {pred}")
            
        return result
        
    except Exception as e:
        print(f"Error during LLM classification: {str(e)}")
        return None

def main():
    """Main function to test both ML and LLM classifiers."""
    print("Loading and preparing data...")
    
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
    
    # Take random 10 rows for training and random 5 for testing
    # First shuffle the dataframe to ensure randomness
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df_shuffled.iloc[:10]
    test_df = df_shuffled.iloc[10:15]
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Test traditional ML classifier (RoBERTa)
    print("\n" + "="*100)
    print("STARTING CLASSIFICATION TESTS")
    print("="*100)
    
    ml_result = test_ml_classifier(train_df, test_df, label_columns)
    
    # Test LLM classifier (DeepSeek) - COMMENTED OUT
    # llm_result = test_llm_classifier(train_df, test_df, label_columns)
    
    # Compare results if both succeeded - COMMENTED OUT
    # if ml_result and llm_result:
    #     print("\n" + "="*80)
    #     print("COMPARISON OF RESULTS")
    #     print("="*80)
    #     print("Side-by-side comparison of ML vs LLM predictions:")
    #     print("-" * 80)
    #     for idx, (text, ml_pred, llm_pred) in enumerate(zip(
    #         test_df['text'], 
    #         ml_result.predictions, 
    #         llm_result.predictions
    #     )):
    #         print(f"\nTest {idx + 1}:")
    #         print(f"Text: {text[:100]}...")
    #         print(f"ML Prediction:  {ml_pred}")
    #         print(f"LLM Prediction: {llm_pred}")
    #         print("-" * 40)

if __name__ == "__main__":
    main()