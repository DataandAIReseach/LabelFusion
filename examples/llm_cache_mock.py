"""Quick mock to test LLMPredictionCache store/load/has/get and skipping logic.

Run: python examples/llm_cache_mock.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so local imports work when running this example
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from importlib.util import spec_from_file_location, module_from_spec
import types

# Load prediction_cache module directly to avoid importing the whole package.
# To allow the file's relative imports (e.g. "from ..core.exceptions import PersistenceError"),
# create minimal parent package entries in sys.modules so the relative import can resolve.
pc_path = project_root / "textclassify" / "llm" / "prediction_cache.py"
module_name = "textclassify.llm.prediction_cache"

# Ensure minimal package modules exist
if 'textclassify' not in sys.modules:
    sys.modules['textclassify'] = types.ModuleType('textclassify')
if 'textclassify.core' not in sys.modules:
    sys.modules['textclassify.core'] = types.ModuleType('textclassify.core')
if 'textclassify.llm' not in sys.modules:
    sys.modules['textclassify.llm'] = types.ModuleType('textclassify.llm')

# Provide a lightweight stub for textclassify.core.exceptions so prediction_cache can import PersistenceError
if 'textclassify.core.exceptions' not in sys.modules:
    exc_mod = types.ModuleType('textclassify.core.exceptions')
    class PersistenceError(Exception):
        pass
    exc_mod.PersistenceError = PersistenceError
    sys.modules['textclassify.core.exceptions'] = exc_mod

spec = spec_from_file_location(module_name, str(pc_path))
pc_mod = module_from_spec(spec)
spec.loader.exec_module(pc_mod)
LLMPredictionCache = pc_mod.LLMPredictionCache
import pandas as pd
import shutil

CACHE_DIR = "cache/mock_llm"

# Clean previous mock cache so this demo is repeatable
try:
    shutil.rmtree(CACHE_DIR)
except Exception:
    pass

# Initialize cache (will create directory)
cache = LLMPredictionCache(cache_dir=CACHE_DIR, verbose=True)
print("Initial cache stats:", cache.get_cache_stats())

# Example texts (note: hashing is text-based)
texts = [
    "The quick brown fox",
    "Jumps over the lazy dog",
    "Hello world",
    "Example text A",
    "Example text B"
]

# Simulate storing predictions for two texts
cache.store_prediction(texts[0], [1,0,0], "resp1", "prompt1")
cache.store_prediction(texts[2], [0,1,0], "resp2", "prompt2")

# Force a save so a subsequent process can discover the cache files
cache.save_cache()

print("After storing 2 predictions, stats:", cache.get_cache_stats())

# Build a test DataFrame that our prediction runner would use
df_test = pd.DataFrame({"text": texts})

# Determine which texts are cached
is_cached = df_test['text'].apply(lambda t: cache.has_prediction(str(t)))
print("Cached mask:", is_cached.tolist())
print("Cached rows:")
print(df_test[is_cached])
print("Uncached rows:")
print(df_test[~is_cached])

# Reinitialize cache object to simulate restart and loading from disk
cache2 = LLMPredictionCache(cache_dir=CACHE_DIR, verbose=True)
print("Reloaded cache stats:", cache2.get_cache_stats())

# Fetch prediction for a cached text
pred = cache2.get_prediction(texts[0])
print("Fetched cached prediction for first text:", pred)

print("Mock complete")
