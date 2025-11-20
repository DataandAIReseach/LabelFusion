"""Precache example for Fusion ensemble (val/test) using few-shot LLM prompts.

This example mirrors the `PRECACHE_LLM` section used in the `eval_reuters.py`
script: it creates a small few-shot training sample, generates LLM predictions
for a validation and a test split, and then uses FusionEnsemble helpers to save
the cached predictions in the same format the ensemble expects.

The script is defensive: it will try to use the real `OpenAIClassifier` or
`DeepSeekClassifier` if available, and otherwise falls back to a `MockLLM`.
Likewise a minimal `MockML` is used for constructing the temporary `FusionEnsemble`.

Run as:
  python3 examples/ensemble_cache_interrupt_mock.py

This will create files under `cache/val_*` and `cache/test_*` (or print
diagnostics if optional packages are missing).
"""

import os
import sys
from pathlib import Path
import random
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Lightweight stub for `dotenv.load_dotenv` when `python-dotenv` is not installed.
# This prevents `ModuleNotFoundError` during imports of package modules that
# call `from dotenv import load_dotenv` at import time.
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
from importlib.util import spec_from_file_location, module_from_spec


CACHE_DIR = 'cache/mock_ensemble'
os.makedirs(CACHE_DIR, exist_ok=True)


def _safe_load_fusion():
    """Load FusionEnsemble, falling back to a source import with minimal stubs."""
    try:
        from textclassify.ensemble.fusion import FusionEnsemble
        return FusionEnsemble
    except Exception:
        # load from source and provide tiny stubs required by fusion.py
        fusion_path = project_root / 'textclassify' / 'ensemble' / 'fusion.py'
        spec = spec_from_file_location('textclassify.ensemble.fusion', str(fusion_path))
        mod = module_from_spec(spec)

        # Create minimal stubs used by fusion.py if not present
        import types as _types, sys as _sys
        if 'textclassify.core.types' not in _sys.modules:
            ct = _types.ModuleType('textclassify.core.types')
            class EnsembleConfig:
                def __init__(self, **kwargs):
                    self.parameters = kwargs.get('parameters', {})
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


class MockLLM:
    def __init__(self, classes=None):
        self.classes_ = classes or ['A', 'B', 'C']
        self.text_column = 'text'

    def predict(self, train_df=None, test_df=None):
        # Return one predicted label per test row (simple random choice)
        preds = [[random.choice(self.classes_)] for _ in range(len(test_df))]
        return type('R', (), {'predictions': preds, 'metadata': {}})()


class MockML:
    def __init__(self):
        self.text_column = 'text'
        self.label_columns = []


def main():
    # create tiny synthetic val/test sets to exercise the caching helper
    texts_val = [f"val sample {i}" for i in range(5)]
    texts_test = [f"test sample {i}" for i in range(5)]
    df_val = pd.DataFrame({'text': texts_val})
    df_test = pd.DataFrame({'text': texts_test})

    few_shot = 2
    # sample few-shot training context
    train_small = pd.DataFrame({'text': [f"few shot {i}" for i in range(few_shot)], 'labels': [0]*few_shot})

    # Try to create a real LLM classifier (OpenAI/DeepSeek) if available, otherwise fallback
    llm = None
    try:
        from textclassify.llm.openai_classifier import OpenAIClassifier
        from textclassify.core.types import ModelConfig, ModelType
        cfg = ModelConfig(model_name='gpt-5-nano', model_type=ModelType.LLM, parameters={})
        llm = OpenAIClassifier(config=cfg, text_column='text', label_columns=['labels'], multi_label=True,
                                few_shot_mode=few_shot, auto_use_cache=True, cache_dir=CACHE_DIR,
                                auto_save_results=False)
        print('Using real OpenAIClassifier')
    except Exception:
        try:
            from textclassify.llm.deepseek_classifier import DeepSeekClassifier
            # deepseek usage may differ — if available, try a minimal init
            llm = DeepSeekClassifier(config=None, text_column='text', label_columns=['labels'], multi_label=True,
                                     few_shot_mode=few_shot, auto_save_results=False, auto_use_cache=True,
                                     cache_dir=CACHE_DIR)
            print('Using real DeepSeekClassifier')
        except Exception:
            print('Real LLM not available — using MockLLM fallback')
            llm = MockLLM()

    # Generate predictions for val and test using few-shot context
    print('Generating validation predictions...')
    val_result = llm.predict(train_df=train_small, test_df=df_val)
    print(f'  -> {len(val_result.predictions)} validation preds generated')

    print('Generating test predictions...')
    test_result = llm.predict(train_df=train_small, test_df=df_test)
    print(f'  -> {len(test_result.predictions)} test preds generated')

    # Create a temporary ML + Fusion ensemble to write cache files using the same helpers
    FusionEnsemble = _safe_load_fusion()
    # Simple config object expected by FusionEnsemble. Provide the minimal
    # attributes the constructor reads (ensemble_method, parameters, models).
    class SimpleConfig:
        def __init__(self, parameters=None):
            self.parameters = parameters or {}
            # FusionEnsemble checks ensemble_config.ensemble_method in some codepaths
            self.ensemble_method = 'fusion'
            # Provide empty models list to avoid attribute errors elsewhere
            self.models = []
            # Ensure a few common parameter defaults exist so FusionEnsemble can
            # read paths without falling back to external inputs.
            if 'output_dir' not in self.parameters:
                self.parameters['output_dir'] = 'outputs/mock_ensemble'
            if 'experiment_name' not in self.parameters:
                self.parameters['experiment_name'] = 'precache_example'
            if 'auto_save_results' not in self.parameters:
                self.parameters['auto_save_results'] = False

    params = {'val_llm_cache_path': os.path.join(CACHE_DIR, 'val'), 'test_llm_cache_path': os.path.join(CACHE_DIR, 'test')}
    cfg = SimpleConfig(parameters=params)
    # Populate `models` so BaseEnsemble validation passes. Use the mock ML and the
    # LLM instance we already created. These are placeholders; real model objects
    # will be attached properly below via `add_ml_model` / `add_llm_model`.
    try:
        cfg.models = [MockML(), llm]
    except Exception:
        cfg.models = [1]  # fallback non-empty marker
    # Provide a couple of flags expected by BaseEnsemble/model_info
    cfg.require_all_models = False
    cfg.fallback_model = None

    try:
        fusion = FusionEnsemble(cfg, output_dir='outputs/mock_ensemble', experiment_name='precache_example', auto_save_results=False)
    except Exception as e:
        print(f'Could not instantiate FusionEnsemble: {e}')
        print('Aborting cache save step')
        return

    # Attach a mock ML (not used for saving LLM-only caches)
    fusion.add_ml_model(MockML())

    # Persist cached LLM predictions using FusionEnsemble helper
    os.makedirs(os.path.join(CACHE_DIR, 'val'), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, 'test'), exist_ok=True)

    try:
        fusion._save_cached_llm_predictions(val_result.predictions, os.path.join(CACHE_DIR, 'val'), df_val)
        fusion._save_cached_llm_predictions(test_result.predictions, os.path.join(CACHE_DIR, 'test'), df_test)
        print('✅ Precache complete. Cached files saved under cache/val_* and cache/test_*')
    except Exception as e:
        print(f'Failed to save cached predictions: {e}')


if __name__ == '__main__':
    main()
"""Simulate ensemble LLM prediction with incremental caching and an abrupt cancel.

This script runs a single demo that:
 - starts incremental LLM prediction and simulates an abrupt cancellation
 - shows partial cache on disk
 - resumes prediction and completes missing entries

It intentionally avoids heavy optional dependencies; it loads
`textclassify/llm/prediction_cache.py` directly so it can run in minimal envs.
"""

import sys
from pathlib import Path
import types
import time
import random
import math

# Project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from importlib.util import spec_from_file_location, module_from_spec

# Prepare minimal package stubs so relative imports in prediction_cache work
if 'textclassify' not in sys.modules:
    sys.modules['textclassify'] = types.ModuleType('textclassify')
if 'textclassify.core' not in sys.modules:
    sys.modules['textclassify.core'] = types.ModuleType('textclassify.core')
if 'textclassify.llm' not in sys.modules:
    sys.modules['textclassify.llm'] = types.ModuleType('textclassify.llm')
if 'textclassify.core.exceptions' not in sys.modules:
    exc_mod = types.ModuleType('textclassify.core.exceptions')
    class PersistenceError(Exception):
        pass
    exc_mod.PersistenceError = PersistenceError
    sys.modules['textclassify.core.exceptions'] = exc_mod

# Load the prediction_cache module directly
pc_path = project_root / 'textclassify' / 'llm' / 'prediction_cache.py'
spec = spec_from_file_location('textclassify.llm.prediction_cache', str(pc_path))
pc_mod = module_from_spec(spec)
spec.loader.exec_module(pc_mod)
LLMPredictionCache = pc_mod.LLMPredictionCache

# Do not import FusionEnsemble at top-level (examples may run with package stubs).
# We'll load the module dynamically later when needed.

import pandas as pd
import os

CACHE_DIR = 'cache/mock_ensemble'


def make_texts(n=10):
    return [f"Sample text #{i} - {random.choice(['alpha','beta','gamma','delta'])}" for i in range(n)]


# `FusionEnsembleStub` removed — not needed for this incremental cache demo.


def run_demo(abort_after_batches: int = 2, batch_size: int = 3):
    # Clean for fresh demo
    try:
        import shutil
        shutil.rmtree(CACHE_DIR)
    except Exception:
        pass
    # Prepare DataFrame
    texts = make_texts(10)
    df = pd.DataFrame({'text': texts})

    # We'll demonstrate the incremental LLM prediction via FusionEnsemble.
    # Create a minimal ensemble config object that FusionEnsemble expects.
    class SimpleEnsembleConfig:
        def __init__(self, parameters=None):
            self.parameters = parameters or {}

    # Minimal mock ML model -- only predict/predict_without_saving required by FusionEnsemble
    class MockML:
        def __init__(self):
            self.text_column = 'text'
