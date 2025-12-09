LLM Demos and Cache Documentation

Overview

This document describes the lightweight runnable demos added under `examples/`.
They are intentionally safe to run in minimal environments: each demo tries to
use the real library classes (OpenAIClassifier, FusionEnsemble, AutoFusion, etc.)
and falls back to simple mocks when those classes or credentials are not
available.

Files of interest (new/renamed)

- `examples/cache_usage_demo.py` — Demonstrates writing and discovering simple
  LLM cache JSON files. Writes a small mock cache under `cache/demo_llm_cache/`.

- `examples/ensemble_cache_interrupt_demo.py` — Creates small val/test sets and
  writes simple mock cache files under `cache/mock_ensemble/val` and
  `cache/mock_ensemble/test`.

- `examples/llm_cache_mock.py` — Existing example that exercises the
  `LLMPredictionCache` implementation. If the real `prediction_cache` module is
  available it will use it; otherwise it will raise at import-time. This demo
  is left as an integration-oriented example.

- `examples/minimal_precache_demo.py` — Minimal precache flow that tries to use
  a real LLM (OpenAI/DeepSeek) and falls back to `MockLLM`. It uses Fusion helper
  `_save_cached_llm_predictions` to write canonical cache JSON files.

- `examples/test_multilabel_autofusion.py` — Runnable multi-label AutoFusion
  demo; falls back to `MockAutoFusion` if `AutoFusionClassifier` is unavailable.

- `examples/test_singlelabel_ml.py` — Runnable single-label ML-only demo; uses
  `RoBERTaClassifier` if available, otherwise `MockML`.

- `examples/test_singlelabel_autofusion.py` — Runnable single-label AutoFusion
  demo with a safe fallback.

Why these demos exist

- Provide quick examples for contributors to run locally without needing
  expensive GPU access or API credentials.
- Demonstrate cache file formats and helper functions for saving and
  discovering cached LLM predictions.
- Provide reproducible scripted flows for CI smoke checks (syntax + import).

How to run (quick)

Run a single demo with Python:

```bash
python examples/test_singlelabel_ml.py
python examples/cache_usage_demo.py
python examples/test_multilabel_autofusion.py
```

Running under a minimal environment will activate mock fallbacks — this
ensures the demos are useful even without model weights or API keys.

Cache helper summary

- `LLMPredictionCache` (in `textclassify/llm/prediction_cache.py`) provides
  low-level operations to store, find, and load cached predictions.
- Fusion helpers (e.g. `_save_cached_llm_predictions`) save canonical JSON files
  used by the ensemble utilities. The demo scripts call these helpers when
  available and otherwise write simple JSON files with a `predictions` list.

Notes for maintainers

- Keep the demos import-safe: avoid executing heavy logic at module import time.
  Use `if __name__ == '__main__'` guards (already applied in the demos).
- The demos intentionally keep output minimal and use `random` to generate
  deterministic-like behavior for quick inspection.
- If you want a CI job that verifies the demos, add a basic step that runs
  `python -m py_compile examples/*.py` and optionally executes a small subset
  with `python -c 'import runpy; runpy.run_path("examples/test_singlelabel_ml.py")'`.

FAQ

Q: Do the demos require API keys?
A: No — they fall back to mock implementations unless you configure real
   credentials and install optional dependencies.

Q: Where are cache files written?
A: `cache/` under the repository root. Demo scripts use subfolders like
   `cache/demo_llm_cache` and `cache/mock_ensemble`.

Q: Should we commit cache files?
A: No — cache files are runtime artifacts and should remain untracked. Add
   them to `.gitignore` if they are not already ignored.

If you'd like, I can also add a short CI job (GitHub Actions) that runs the
syntax checks and executes one demo with mocks. Say the word and I'll add it.