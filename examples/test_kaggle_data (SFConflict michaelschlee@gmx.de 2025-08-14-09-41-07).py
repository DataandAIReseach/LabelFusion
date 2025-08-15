"""Test for both traditional ML and LLM text classification."""

import os
import pandas as pd
from dotenv import load_dotenv
import asyncio
from textclassify.llm.deepseek_classifier import DeepSeekClassifier
from textclassify.llm.gemini_classifier import GeminiClassifier  # Gemini alternative
from textclassify.llm.openai_classifier import OpenAIClassifier  # OpenAI alternative
from textclassify.ml.roberta_classifier import RoBERTaClassifier  # Traditional ML classifier
from textclassify.config.settings import Config
from textclassify.core.types import ModelType, TrainingData, ModelConfig, ClassificationType

# Load environment variables
load_dotenv()

def test_ml_classifier(train_df, test_df, label_columns):
    """Test the traditional RoBERTa ML classifier."""
    print("\n" + "="*80)
    print("TESTING TRADITIONAL ML CLASSIFIER (RoBERTa)")
    print("="*80)
    
    # Create ML configuration
    config = ModelConfig(
        model_name='roberta-base',
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            'model_name': 'roberta-base',
            'max_length': 512,
            'learning_rate': 2e-5,
            'epochs': 3,
            'batch_size': 8,
            'warmup_steps': 100,
            'label_columns': label_columns  # Add label names for proper class mapping
        }
    )
    
    # Create training data for ML classifier
    training_data = TrainingData(
        texts=train_df['text'].tolist(),
        labels=train_df[label_columns].values.tolist(),
        classification_type=ClassificationType.MULTI_CLASS
    )
    
    # Initialize ML classifier
    ml_classifier = RoBERTaClassifier(config=config)
    
    try:
        # Train the classifier
        print("Training RoBERTa classifier...")
        ml_classifier.fit(training_data)
        
        # Make predictions and calculate metrics
        print("Making predictions and calculating metrics...")
        test_texts = test_df['text'].tolist()
        true_labels = test_df[label_columns].values.tolist()
        
        # Use predict method with true labels to get predictions with metrics
        result = ml_classifier.predict(test_texts, true_labels)
        
        # Print metrics from result object
        metrics = result.metadata.get('metrics', {})
        if ml_classifier.classification_type == ClassificationType.MULTI_CLASS:
            print(f"\nMulti-class Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"Weighted Precision: {metrics.get('precision_weighted', 'N/A'):.4f}")
            print(f"Weighted Recall: {metrics.get('recall_weighted', 'N/A'):.4f}")
            print(f"Weighted F1-Score: {metrics.get('f1_weighted', 'N/A'):.4f}")
        else:
            print(f"\nMulti-label Exact Match Accuracy: {metrics.get('exact_match_accuracy', 'N/A'):.4f}")
            print(f"Hamming Loss: {metrics.get('hamming_loss', 'N/A'):.4f}")
            print(f"Weighted Precision: {metrics.get('precision_weighted', 'N/A'):.4f}")
            print(f"Weighted Recall: {metrics.get('recall_weighted', 'N/A'):.4f}")
            print(f"Weighted F1-Score: {metrics.get('f1_weighted', 'N/A'):.4f}")
        
        # Print results
        print("\nML Classifier Results:")
        print("-" * 60)
        for idx, (text, pred, true_label) in enumerate(zip(test_df['text'], result.predictions, true_labels)):
            print(f"\nTest {idx + 1}:")
            print(f"Text: {text[:100]}...")
            print(f"True Label: {[label_columns[i] for i, val in enumerate(true_label) if val == 1]}")
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