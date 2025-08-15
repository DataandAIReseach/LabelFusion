# LLM+ML Fusion Ensemble

This module provides a sophisticated fusion framework that combines traditional Machine Learning models (like RoBERTa) with Large Language Models (LLMs) using a trainable neural network to achieve superior text classification performance.

## 🎯 Super Simple Interface - AutoFusion

**NEW**: For users who want maximum performance with minimal effort, use the `AutoFusionClassifier`:

```python
from textclassify import AutoFusionClassifier

# This is ALL you need! 🚀
config = {
    'llm_provider': 'deepseek',  # Choose: 'deepseek', 'openai', or 'gemini'
    'label_columns': ['positive', 'negative', 'neutral']
}

classifier = AutoFusionClassifier(config)
classifier.fit(your_dataframe)  # Automatic ML+LLM fusion training
result = classifier.predict(test_texts)  # Get superior predictions
```

Everything else (ML model setup, LLM configuration, fusion training, calibration) happens automatically behind the scenes!

## Overview

The Fusion Ensemble works by:

1. **ML Component**: Uses your existing RoBERTa classifier to generate logits from text input
2. **LLM Component**: Uses your existing LLM classifiers (DeepSeek, OpenAI, Gemini) to generate class scores
3. **Fusion MLP**: A trainable neural network that learns to optimally combine ML logits and LLM scores
4. **Training Strategy**: Uses different learning rates for different components (small LR for ML backbone, higher LR for fusion MLP)
5. **Calibration**: Includes LLM score calibration for improved probability estimates

## Key Features

- **🎯 Zero-Hassle Interface**: Just specify LLM provider - everything else is automatic
- **Reuses Existing Models**: No need to reimplement - leverages your existing RoBERTa and LLM classifiers
- **Multi-Class & Multi-Label**: Supports both classification types seamlessly  
- **Smart Training**: Different learning rates for different components
- **Calibration**: Automatic LLM score calibration using validation data
- **Comprehensive Evaluation**: Built-in metrics calculation and storage
- **Easy Integration**: Works with your existing ensemble framework

## Quick Start

### 🎯 Method 1: AutoFusion (Recommended for most users)

```python
from textclassify import AutoFusionClassifier
import pandas as pd

# Your data
df = pd.DataFrame({
    'text': ["I love this!", "This is terrible", "It's okay"],
    'sentiment': ['positive', 'negative', 'neutral']
})

# Super simple configuration
config = {
    'llm_provider': 'deepseek',  # or 'openai', 'gemini'
    'label_columns': ['sentiment']
}

# That's it! Everything else is automatic
classifier = AutoFusionClassifier(config)
classifier.fit(df)
predictions = classifier.predict(["This is amazing!"])
print(predictions.predictions[0])  # 'positive'
```

### 🏷️ Multi-Label with AutoFusion

```python
# Multi-label is just as simple
config = {
    'llm_provider': 'deepseek',
    'label_columns': ['action', 'comedy', 'drama', 'horror', 'romance'],
    'multi_label': True  # Just set this to True!
}

classifier = AutoFusionClassifier(config)
classifier.fit(movie_dataframe)
result = classifier.predict(["A funny action movie"])
print(result.predictions[0])  # ['action', 'comedy']
```

### ⚙️ Method 2: Advanced Manual Configuration

```python
from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.deepseek_classifier import DeepSeekClassifier
from textclassify.core.types import EnsembleConfig, TrainingData, ClassificationType

# Create your ML and LLM models (using existing implementations)
ml_model = RoBERTaClassifier(config=ml_config)
llm_model = DeepSeekClassifier(config=llm_config, label_columns=labels)

# Create fusion ensemble
fusion_config = EnsembleConfig(
    ensemble_method='fusion',
    models=[ml_model, llm_model],
    parameters={
        'fusion_hidden_dims': [64, 32],
        'ml_lr': 1e-5,           # Small LR for ML backbone
        'fusion_lr': 1e-3,       # Higher LR for fusion MLP
        'num_epochs': 10,
        'batch_size': 16
    }
)

fusion_ensemble = FusionEnsemble(fusion_config)
fusion_ensemble.add_ml_model(ml_model)
fusion_ensemble.add_llm_model(llm_model)

# Train
training_data = TrainingData(texts=texts, labels=labels, classification_type=ClassificationType.MULTI_CLASS)
fusion_ensemble.fit(training_data)

# Predict with automatic metrics calculation
result = fusion_ensemble.predict(test_texts, test_labels)
print(f"Accuracy: {result.metadata['metrics']['accuracy']:.4f}")
```

### 2. Command Line Training

```bash
# Create default configuration
python train_fusion.py --create-config fusion_config.yaml

# Train with configuration file
python train_fusion.py --config fusion_config.yaml

# Quick training with command line args
python train_fusion.py \
    --data data/train.csv \
    --ml-model roberta-base \
    --llm-provider deepseek \
    --llm-model deepseek-chat \
    --output fusion_output

# Training with evaluation
python train_fusion.py \
    --config fusion_config.yaml \
    --test-data data/test.csv
```

### 3. Configuration File

```yaml
# fusion_config.yaml
data:
  file_path: "data/train.csv"
  text_column: "text"
  label_columns: ["positive", "neutral", "negative"]
  multi_label: false

models:
  ml:
    model_name: "roberta-base"
    max_length: 512
    learning_rate: 2e-5
    num_epochs: 3
    batch_size: 16

  llm:
    provider: "deepseek"
    model: "deepseek-chat"
    temperature: 0.1
    max_completion_tokens: 150

fusion:
  hidden_dims: [64, 32]
  ml_learning_rate: 1e-5
  fusion_learning_rate: 1e-3
  num_epochs: 10
  batch_size: 16

output_dir: "fusion_output"
```

## Architecture Details

### Fusion MLP Architecture

```python
class FusionMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32]):
        # input_dim = num_labels * 2  (ML logits + LLM scores)
        # Multiple hidden layers with ReLU and Dropout
        # Output layer for final classification
```

### Fusion Wrapper

```python
class FusionWrapper(nn.Module):
    def forward(self, input_ids, attention_mask, llm_scores):
        # 1. Get ML logits from existing RoBERTa model
        ml_logits = self.ml_model(input_ids, attention_mask)
        
        # 2. Ensure LLM scores are detached (no gradient flow)
        llm_scores = llm_scores.detach()
        
        # 3. Concatenate and pass through fusion MLP
        fusion_input = torch.cat([ml_logits, llm_scores], dim=1)
        fused_logits = self.fusion_mlp(fusion_input)
        
        return {"ml_logits": ml_logits, "llm_scores": llm_scores, "fused_logits": fused_logits}
```

## Training Process

1. **ML Model Training**: Train or use pre-trained RoBERTa model
2. **LLM Score Generation**: Generate LLM scores for training data (with caching)
3. **Data Splitting**: Split data for fusion training and validation  
4. **Fusion Training**: Train fusion MLP with different learning rates
5. **Calibration**: Calibrate LLM scores on validation data
6. **Evaluation**: Calculate comprehensive metrics

## Multi-Label Support

The fusion ensemble automatically handles multi-label classification:

```python
# Multi-label configuration
training_data = TrainingData(
    texts=texts,
    labels=binary_labels,  # [[1,0,1], [0,1,0], ...]
    classification_type=ClassificationType.MULTI_LABEL
)

# Fusion ensemble automatically adapts
fusion_ensemble.fit(training_data)

# Get multi-label predictions
result = fusion_ensemble.predict(test_texts, test_labels)
# result.predictions = [['action', 'drama'], ['comedy'], ...]
```

## Utilities and Helpers

```python
from textclassify.utils.fusion_utils import FusionUtils

# Data preparation
train_df, test_df = FusionUtils.prepare_data_for_fusion(
    df, text_column='text', label_columns=['label'], multi_label=False
)

# Data validation
stats = FusionUtils.validate_fusion_data(df, 'text', ['label'])
print(f"Data valid: {stats['valid']}")
print(f"Text stats: {stats['statistics']['text']}")

# Model evaluation
evaluation = FusionUtils.evaluate_fusion_model(
    fusion_ensemble, test_texts, test_labels, save_results=True
)

# Visualizations (requires matplotlib)
FusionUtils.plot_fusion_results(evaluation, save_path='plots/')

# Model persistence
FusionUtils.save_fusion_model(fusion_ensemble, 'models/fusion_v1/')
loaded_model = FusionUtils.load_fusion_model('models/fusion_v1/')
```

## Advanced Features

### Custom Fusion MLP Architecture

```python
fusion_config = EnsembleConfig(
    ensemble_method='fusion',
    parameters={
        'fusion_hidden_dims': [128, 64, 32],  # Deeper network
        'ml_lr': 5e-6,                        # Even smaller ML LR
        'fusion_lr': 2e-3,                    # Higher fusion LR  
        'num_epochs': 15
    }
)
```

### LLM Score Calibration

The fusion ensemble automatically calibrates LLM scores:
- **Multi-class**: Temperature scaling using logistic regression
- **Multi-label**: Isotonic regression per label

### Caching

LLM scores are automatically cached to disk to avoid recomputation:
- Cache key based on input texts
- Persistent across training runs
- Significant speedup for repeated experiments

## Examples

See the `examples/` directory for complete working examples:

- `fusion_ensemble_example.py`: Basic multi-class and multi-label examples
- `train_fusion.py`: Full command-line training script
- `config/fusion_config.yaml`: Example configuration file

## Requirements

- Python 3.8+
- PyTorch
- transformers  
- scikit-learn
- numpy
- pandas
- pyyaml
- matplotlib (optional, for plotting)
- seaborn (optional, for plotting)

## Integration with Existing Code

The fusion ensemble seamlessly integrates with your existing ensemble framework:

```python
# Works with other ensemble methods
from textclassify.ensemble import VotingEnsemble, WeightedEnsemble, FusionEnsemble

# All ensembles follow the same interface
ensembles = [
    VotingEnsemble(voting_config),
    WeightedEnsemble(weighted_config), 
    FusionEnsemble(fusion_config)
]

for ensemble in ensembles:
    ensemble.fit(training_data)
    result = ensemble.predict(test_texts, test_labels)
    print(f"{type(ensemble).__name__}: {result.metadata['metrics']['accuracy']:.4f}")
```

## Best Practices

1. **Learning Rates**: Use small LR (1e-5) for ML backbone, higher LR (1e-3) for fusion MLP
2. **Data Splits**: Reserve 20% of training data for fusion training and calibration
3. **Validation**: Always validate on held-out test data not used for training
4. **Caching**: Enable LLM score caching for faster experimentation
5. **Calibration**: Use calibrated scores for better probability estimates
6. **Architecture**: Start with [64, 32] hidden dims, adjust based on data size

## Troubleshooting

### Common Issues

1. **CUDA Memory**: Reduce batch_size if running out of GPU memory
2. **LLM API Limits**: Use smaller batch sizes and enable retries
3. **Training Instability**: Reduce learning rates, especially for ML backbone
4. **Poor Calibration**: Ensure validation set is representative of test data

### Performance Tips

1. Use GPU acceleration for faster training
2. Enable LLM score caching to avoid API recomputation  
3. Use mixed precision training for large models
4. Batch LLM API calls for efficiency

## Contributing

The fusion ensemble is designed to be extensible. To add new fusion architectures:

1. Extend `FusionMLP` class for new architectures
2. Modify `FusionWrapper` for different combination strategies  
3. Add new calibration methods in the calibration pipeline
4. Update configuration schema for new parameters

## License

This fusion ensemble implementation is part of the ClassifyFusion package and follows the same license terms.
