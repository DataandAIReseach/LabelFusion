#!/usr/bin/env python3
import os
print('Starting LLM precache script')
from tests.evaluation.eval_reuters import load_datasets, create_llm_model, create_ml_model, create_fusion_ensemble

print('Loading datasets...')
df_train, df_val, df_test = load_datasets()

few_shot = int(os.environ.get('FEW_SHOT', '20'))
train_small = df_train.sample(n=min(few_shot, len(df_train)), random_state=42)

# Determine label columns heuristically (exclude text)
label_cols = [c for c in df_train.columns if c != 'text']
print(f'Sampled {len(train_small)} examples for few-shot context; label cols: {label_cols}')

llm = create_llm_model('text', label_cols, provider='openai', output_dir='outputs/reuters_availability_experiments', experiment_name='precache', cache_dir='cache', few_shot_examples=few_shot)
print('Created LLM instance; generating and saving val/test predictions separately...')

# Generate and save validation predictions
print('-> Generating validation predictions...')
val_result = llm.predict(train_df=train_small, test_df=df_val)
val_preds = val_result.predictions
print(f'  Generated {len(val_preds)} validation predictions')

# Generate and save test predictions
print('-> Generating test predictions...')
test_result = llm.predict(train_df=train_small, test_df=df_test)
test_preds = test_result.predictions
print(f'  Generated {len(test_preds)} test predictions')

# Use FusionEnsemble helper to save cache files with the expected naming/hash structure
print('Saving predictions into fusion cache format...')
ml_model = create_ml_model('text', label_cols, 'outputs/reuters_availability_experiments', 'precache_ml')
fusion = create_fusion_ensemble(ml_model, llm, 'outputs/reuters_availability_experiments', 'precache_fusion', cache_dir='cache')
fusion._save_cached_llm_predictions(val_preds, os.path.join('cache', 'val'), df_val)
fusion._save_cached_llm_predictions(test_preds, os.path.join('cache', 'test'), df_test)

print('Precache complete. Cache files saved under cache/val* and cache/test*')
