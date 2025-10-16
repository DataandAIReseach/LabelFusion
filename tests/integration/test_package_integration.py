import pandas as pd
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.core.types import ModelConfig, ModelType, EnsembleConfig

# Load your dataset (same as RoBERTa test)

train_df = pd.read_csv("/user/schlee2/u19147/repos/LabelFusion/data/ag_news/ag_train_balanced.csv")
val_df = pd.read_csv("/user/schlee2/u19147/repos/LabelFusion/data/ag_news/ag_val_balanced.csv")
test_df = pd.read_csv("/user/schlee2/u19147/repos/LabelFusion/data/ag_news/ag_test_balanced.csv")

# Configure ML model (RoBERTa)
ml_config = ModelConfig(
model_name="roberta-base",
model_type=ModelType.TRADITIONAL_ML,
parameters={
    "model_name": "roberta-base",
    "learning_rate": 2e-5,
    "num_epochs": 2,  # Fast training for testing
    "batch_size": 8,
    "max_length": 256
}
)

# Create RoBERTa classifier - disable auto_save_results to prevent standalone directories
ml_classifier = RoBERTaClassifier(
config=ml_config,
text_column='description',
label_columns=["label_1", "label_2", "label_3", "label_4"],
multi_label=False,
auto_save_path="cache/experimente/fusion_roberta_model",
auto_save_results=True  # Prevent standalone ML experiment directory
)

# Configure LLM model (OpenAI)
llm_config = ModelConfig(
model_name="gpt-5-nano",
model_type=ModelType.LLM,
parameters={
    "model": "gpt-5-nano",
    "temperature": 0.1,
    "max_completion_tokens": 150,
    "top_p": 1.0
}
)

# Create OpenAI LLM classifier - disable auto_save_results to prevent standalone directories
llm_classifier = OpenAIClassifier(
config=llm_config,
text_column='description',
label_columns=["label_1", "label_2", "label_3", "label_4"],
enable_cache=True,
cache_dir="cache/experimente/fusion_openai_cache",
multi_label=False,
auto_save_results=True  # Prevent standalone LLM experiment directory
)

# Configure Fusion Ensemble
fusion_config = EnsembleConfig(
ensemble_method="fusion",
models=[ml_classifier, llm_classifier],
parameters={
    "fusion_hidden_dims": [64, 32],  # Smaller network to prevent overfitting
    "ml_lr": 1e-5,  # Low LR for stable RoBERTa fine-tuning
    "fusion_lr": 5e-4,  # Lower LR for more stable fusion training
    "num_epochs": 10,  # Fewer epochs (early stopping at ~6-7 would be ideal)
    "batch_size": 8,
    "classification_type": "multi_class",
    "val_llm_cache_path": "cache/fusion_openai_cache/val",
    "test_llm_cache_path": "cache/fusion_openai_cache/test"
}
)

# Create Fusion Ensemble - this will save ONLY the fusion results, not intermediate LLM predictions
fusion_ensemble = FusionEnsemble(
    fusion_config, 
    auto_save_results=True,
    save_intermediate_llm_predictions=False,  # Disable intermediate LLM prediction saving
    auto_use_cache=True  # Enable automatic cache usage for LLM predictions
)

# Add models to ensemble
fusion_ensemble.add_ml_model(ml_classifier)
fusion_ensemble.add_llm_model(llm_classifier)

# Use small samples for testing (same as RoBERTa test)
#train_sample = train_df.sample(50, random_state=42)  # Zufällig 50 Zeilen
#val_sample = val_df.sample(5, random_state=42)       # Zufällig 5 Zeilen
#test_sample = test_df.sample(5, random_state=42)     # Zufällig 5 Zeilen

# Train the fusion ensemble (like RoBERTa fit method)
training_result = fusion_ensemble.fit(train_df, val_df)

# Test prediction (like RoBERTa predict method)
result = fusion_ensemble.predict(test_df=test_df)