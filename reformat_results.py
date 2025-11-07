"""
Script to reformat the existing results with correct metric names.
"""
import json
import pandas as pd
from datetime import datetime

# Load the detailed results
with open('outputs/data_availability_experiments/detailed_results_20251020_200045.json', 'r') as f:
    all_results = json.load(f)

# Reformat summary data with correct metric names
summary_data = []
for result in all_results:
    # Add fusion results
    # Fusion uses 'f1', 'precision', 'recall' (without suffixes)
    fusion_metrics = result['test_metrics']
    summary_data.append({
        'percentage': f"{int(result['percentage']*100)}%",
        'model': 'Fusion',
        'train_samples': result['train_samples'],
        'accuracy': fusion_metrics.get('accuracy', 0.0),
        'f1': fusion_metrics.get('f1', 0.0),
        'precision': fusion_metrics.get('precision', 0.0),
        'recall': fusion_metrics.get('recall', 0.0)
    })
    
    # Add baseline results if available
    if 'all_models' in result:
        all_models = result['all_models']
        
        if 'roberta' in all_models:
            # RoBERTa uses 'f1_weighted', 'precision_weighted', 'recall_weighted'
            ml_metrics = all_models['roberta']['test_metrics']
            summary_data.append({
                'percentage': f"{int(result['percentage']*100)}%",
                'model': 'RoBERTa',
                'train_samples': result['train_samples'],
                'accuracy': ml_metrics.get('accuracy', 0.0),
                'f1': ml_metrics.get('f1_weighted', 0.0),
                'precision': ml_metrics.get('precision_weighted', 0.0),
                'recall': ml_metrics.get('recall_weighted', 0.0)
            })
        
        if 'llm' in all_models:
            # LLM uses 'f1', 'precision', 'recall' (without suffixes)
            llm_metrics = all_models['llm']['test_metrics']
            summary_data.append({
                'percentage': f"{int(result['percentage']*100)}%",
                'model': 'OPENAI',
                'train_samples': result['train_samples'],
                'accuracy': llm_metrics.get('accuracy', 0.0),
                'f1': llm_metrics.get('f1', 0.0),
                'precision': llm_metrics.get('precision', 0.0),
                'recall': llm_metrics.get('recall', 0.0)
            })

summary_df = pd.DataFrame(summary_data)
print("="*80)
print("REFORMATTED SUMMARY WITH CORRECT METRICS")
print("="*80)
print("\n" + summary_df.to_string(index=False))

# Save corrected summary
output_file = f"outputs/data_availability_experiments/summary_corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
summary_df.to_csv(output_file, index=False)
print(f"\n✅ Corrected summary saved to: {output_file}")
