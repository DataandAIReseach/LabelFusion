"""Minimal precache demo based on eval_reuters.py precache block.

- Builds tiny val/test DataFrames
- Tries real LLM (OpenAI/DeepSeek) and falls back to MockLLM
- Uses FusionEnsemble helpers to write canonical cache JSON files

Run:
  python3 examples/minimal_precache_demo.py
"""

import os
import sys
from pathlib import Path
import random
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Safe dotenv stub for minimal envs
try:
    import dotenv  # type: ignore
except Exception:
    import types as _types, sys as _sys
    if 'dotenv' not in _sys.modules:
        _dotenv = _types.ModuleType('dotenv')
        def _load_dotenv(*args, **kwargs):
            return False
        _dotenv.load_dotenv = _load_dotenv
        _sys.modules['dotenv'] = _dotenv

import pandas as pd

CACHE_DIR = 'cache/minimal_precache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Minimal helper to load FusionEnsemble from source if package import fails
from importlib.util import spec_from_file_location, module_from_spec

def _safe_load_fusion():
    try:
        from textclassify.ensemble.fusion import FusionEnsemble
        return FusionEnsemble
    except Exception:
        fusion_path = project_root / 'textclassify' / 'ensemble' / 'fusion.py'
        spec = spec_from_file_location('textclassify.ensemble.fusion', str(fusion_path))
        mod = module_from_spec(spec)
        import types as _types, sys as _sys

        # Provide tiny stubs expected by fusion.py
        if 'textclassify.core.types' not in _sys.modules:
            ct = _types.ModuleType('textclassify.core.types')
            class EnsembleConfig:
                def __init__(self, **kwargs):
                    self.parameters = kwargs.get('parameters', {})
                    self.ensemble_method = kwargs.get('ensemble_method', 'fusion')
                    self.models = kwargs.get('models', [])
                    self.require_all_models = kwargs.get('require_all_models', False)
                    self.fallback_model = kwargs.get('fallback_model', None)
            ct.EnsembleConfig = EnsembleConfig
            _sys.modules['textclassify.core.types'] = ct

        if 'textclassify.ensemble.base' not in _sys.modules:
            eb = _types.ModuleType('textclassify.ensemble.base')
            class BaseEnsemble:
                def __init__(self, ensemble_config, output_dir=None, experiment_name=None, auto_save_results=False):
                    self.models = []
                    self.model_names = []
                def _create_result(self, predictions=None, **kwargs):
                    return _types.SimpleNamespace(predictions=predictions or [], metadata={})
            eb.BaseEnsemble = BaseEnsemble
            _sys.modules['textclassify.ensemble.base'] = eb

        spec.loader.exec_module(mod)
        return mod.FusionEnsemble

# Simple MockLLM like other examples
class MockLLM:
    def __init__(self, classes=None):
        self.classes_ = classes or ['A', 'B', 'C']
        self.text_column = 'text'
        self.multi_label = True
        self.config = type('C', (), {'parameters': {'model': 'mock'}, 'model_type': None})
        self.results_manager = None
        self.provider = 'mock'

    def predict(self, train_df=None, test_df=None, **kwargs):
        preds = [[random.choice(self.classes_)] for _ in range(len(test_df))]
        return type('R', (), {'predictions': preds, 'metadata': {}})()

    def predict_texts(self, texts, true_labels=None):
        preds = [[random.choice(self.classes_)] for _ in texts]
        return type('R', (), {'predictions': preds, 'metadata': {'metrics': {}}})()


def main():
    # tiny deterministic val/test
    df_val = pd.DataFrame({'text': [f'val {i}' for i in range(5)], 'label': [0]*5})
    df_test = pd.DataFrame({'text': [f'test {i}' for i in range(5)], 'label': [0]*5})

    # Try constructing a real LLM; fall back to MockLLM
    llm = None
    try:
        from textclassify.llm.openai_classifier import OpenAIClassifier
        from textclassify.core.types import ModelConfig, ModelType
        cfg = ModelConfig(model_name='gpt-5-nano', model_type=ModelType.LLM, parameters={})
        llm = OpenAIClassifier(config=cfg, text_column='text', label_columns=['label'], multi_label=True,
                                few_shot_mode=2, auto_use_cache=True, cache_dir=CACHE_DIR, auto_save_results=False)
        print('Using OpenAIClassifier')
    except Exception:
        try:
            from textclassify.llm.deepseek_classifier import DeepSeekClassifier
            llm = DeepSeekClassifier(config=None, text_column='text', label_columns=['label'], multi_label=True,
                                     few_shot_mode=2, auto_save_results=False, auto_use_cache=True, cache_dir=CACHE_DIR)
            print('Using DeepSeekClassifier')
        except Exception:
            print('Falling back to MockLLM')
            llm = MockLLM()

    # Generate predictions
    print('Generating validation predictions...')
    val_result = llm.predict(train_df=None, test_df=df_val)
    print(f" -> {len(val_result.predictions)} val preds")

    print('Generating test predictions...')
    test_result = llm.predict(train_df=None, test_df=df_test)
    print(f" -> {len(test_result.predictions)} test preds")

    # Use Fusion helper to save canonical cache JSON
    FusionEnsemble = _safe_load_fusion()

    # Build a minimal EnsembleConfig or use stub
    try:
        from textclassify.core.types import EnsembleConfig
        cfg = EnsembleConfig(ensemble_method='fusion', models=[], parameters={'val_llm_cache_path': os.path.join(CACHE_DIR, 'val'), 'test_llm_cache_path': os.path.join(CACHE_DIR, 'test')})
    except Exception:
        # our stub EnsembleConfig was added in _safe_load_fusion if needed
        from textclassify.core.types import EnsembleConfig
        cfg = EnsembleConfig(parameters={'val_llm_cache_path': os.path.join(CACHE_DIR, 'val'), 'test_llm_cache_path': os.path.join(CACHE_DIR, 'test')})

    fusion = None
    try:
        fusion = FusionEnsemble(cfg, output_dir='outputs/minimal_precache', experiment_name='minimal_precache', auto_save_results=False)
    except Exception as e:
        print('Could not instantiate FusionEnsemble:', e)
        print('Attempting to call _save_cached_llm_predictions via dynamic load...')
        # Fallback: try to load the function directly from fusion module
        fusion_mod_path = project_root / 'textclassify' / 'ensemble' / 'fusion.py'
        spec = spec_from_file_location('fusion_mod', str(fusion_mod_path))
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        tmp_fusion = mod.FusionEnsemble(cfg, output_dir='outputs/minimal_precache', experiment_name='minimal_precache', auto_save_results=False)
        fusion = tmp_fusion

    # attach minimal ML so ensemble has models
    try:
        class MockML:
            def __init__(self):
                self.text_column = 'text'
                self.label_columns = ['label']
                self.classes_ = ['A', 'B', 'C']
                self.is_trained = False
        fusion.add_ml_model(MockML())
        fusion.add_llm_model(llm)
    except Exception:
        pass

    # Ensure cache dirs exist
    os.makedirs(os.path.join(CACHE_DIR, 'val'), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, 'test'), exist_ok=True)

    # Save canonical caches
    try:
        fusion._save_cached_llm_predictions(val_result.predictions, os.path.join(CACHE_DIR, 'val'), df_val)
        fusion._save_cached_llm_predictions(test_result.predictions, os.path.join(CACHE_DIR, 'test'), df_test)
        print('Saved cache JSON files under', CACHE_DIR)
    except Exception as e:
        print('Failed to save cache files:', e)


if __name__ == '__main__':
    main()
