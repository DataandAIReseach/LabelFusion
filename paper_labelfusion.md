---
title: 'LabelFusion: Learning to Fuse LLMs and Transformer Classifiers for Robust Text Classification'
tags:
  - Python
  - Natural Language Processing
  - Text Classification
  - Large Language Models
  - Ensemble Learning
  - Calibration
  - Multi-class
  - Multi-label
authors:
  - name: Christoph Weisser
    affiliation: "1, 2"
  - name: LabelFusion contributors
    affiliation: 1
affiliations:
 - name: Campus-Institut Data Science, Göttingen, Germany
   index: 1
 - name: Centre for Statistics, Georg-August-Universität Göttingen, Germany
   index: 2
date: 15 August 2025
bibliography: paper_labelfusion.bib
---

## Summary

LabelFusion is a fusion ensemble for text classification that learns to combine a traditional transformer-based classifier (e.g., RoBERTa) with one or more Large Language Models (LLMs) to deliver accurate and cost‑aware predictions across multi‑class and multi‑label tasks. The package provides a simple high‑level interface (AutoFusion) that trains the full pipeline end‑to‑end, and a configurable API for advanced users. Under the hood, LabelFusion takes vector signals from an ML backbone (logits) and LLM(s) (per‑class scores), calibrates them, and feeds their concatenation into a small multi‑layer perceptron (MLP) that is trained to produce the final prediction. This learned fusion approach captures complementary strengths of LLM reasoning and transformer efficiency, yielding robust performance across domains while enabling practical trade‑offs between accuracy, latency, and cost.

## Statement of Need

Modern text classification spans diverse scenarios—from sentiment and topic tagging to policy enforcement and routing—often under constraints that vary per deployment (throughput, cost ceilings, data privacy). While transformer classifiers such as BERT/RoBERTa achieve strong supervised performance [@devlin2018bert; @liu2019roberta], frontier LLMs can excel in low‑data, ambiguous, or cross‑domain settings [@openai2023gpt4]. No single model family is uniformly best: LLMs are powerful yet comparatively costly and rate‑limited, whereas fine‑tuned transformers are efficient but may struggle with out‑of‑distribution cases.

LabelFusion addresses this gap by: (1) exposing a minimal “AutoFusion” interface that trains a learned combination of an ML backbone and one or more LLMs; (2) supporting both multi‑class and multi‑label classification; (3) providing calibration of LLM scores and ML logits for better probability estimates; and (4) integrating cleanly with existing ensemble utilities. Researchers and practitioners can therefore leverage LLMs where they add value while retaining the speed and determinism of transformer models.

## State of the Field

Ensembles improve robustness by aggregating diverse predictors [@dietterich2000ensemble; @hansen1990neural]. Mixture‑of‑experts approaches further specialize components and learn to combine their outputs [@jacobs1991adaptive]. In applied NLP, common tools such as scikit‑learn [@pedregosa2011scikit] and Hugging Face Transformers [@wolf2019huggingface] offer strong baselines but do not provide a turnkey, learned fusion of LLMs with supervised transformers. Orchestration frameworks (e.g., LangChain) focus on tool use rather than classification ensembles. LabelFusion contributes a focused, production‑minded implementation of a small learned combiner that operates on calibrated per‑class signals from both model families.

## Functionality and Design

LabelFusion consists of three layers:

- ML component: a RoBERTa‑style classifier produces per‑class logits for input texts.
- LLM component(s): provider‑specific classifiers (OpenAI, Claude, Gemini, DeepSeek) return per‑class scores via prompting. Scores are cached to minimize API calls and cost.
- Fusion component: a compact MLP concatenates ML logits and LLM scores and outputs fused logits. The ML backbone is trained/fine‑tuned with a small learning rate; the fusion MLP uses a higher rate, enabling rapid adaptation without destabilizing the encoder.

Key features:

- Multi‑class and multi‑label support with consistent data structures.
- Calibration of model signals (e.g., temperature/Platt‑style and isotonic techniques) for better probability estimates [@guo2017calibration; @zadrozny2002transforming].
- Caching of LLM responses and batched scoring to reduce cost/latency.
- Command‑line training via `train_fusion.py` and YAML configs; or a minimal AutoFusion API for quick starts.
- Seamless use with other ensembles (e.g., voting/weighted) where LabelFusion can serve as a strong base learner.

### Minimal Example (AutoFusion)

```python
from textclassify import AutoFusionClassifier

config = {
    'llm_provider': 'deepseek',
    'label_columns': ['positive', 'negative', 'neutral']
}

clf = AutoFusionClassifier(config)
clf.fit(train_dataframe)              # trains ML backbone, gathers LLM scores, fits fusion MLP
pred = clf.predict(["This is amazing!"])  # fused prediction
```

### CLI and Configuration

Users can generate a starter config and train via the command line:

- Create config: `python train_fusion.py --create-config fusion_config.yaml`
- Train: `python train_fusion.py --config fusion_config.yaml`
- Optional test data and output artifacts are also supported.

## Quality Control

The repository includes unit and integration tests (see `tests/`) that validate configuration handling, core types, and package integration. Fusion‑specific logic is exercised in examples and the CLI, which run end‑to‑end training with deterministic seeds where applicable. LLM scoring paths implement retries and disk caching; transformer training supports standard sanity checks (overfit a small batch, reduced batch sizes for constrained hardware). Metrics (accuracy/F1, per‑label scores) are computed automatically and stored with run artifacts to facilitate regression tracking.

## Availability and Installation

LabelFusion is distributed as part of the `textclassify` package under the MIT license. The fusion components require Python 3.8+ and common scientific Python dependencies (PyTorch, transformers, scikit‑learn, numpy, pandas, PyYAML). Optional plotting depends on matplotlib/seaborn. Installation and quick‑start snippets are provided in the README and `FUSION_README.md`.

## Impact and Use Cases

Empirically, learned fusion tends to outperform any single model when domains vary or label boundaries are ambiguous, with gains attributable to complementary error profiles. Typical applications include:

- Customer feedback analysis with nuanced multi‑label taxonomies.
- Content moderation where uncertain cases benefit from LLM reasoning while routine items rely on the ML backbone.
- Scientific/article routing across heterogeneous topics.

The approach enables pragmatic cost control (e.g., using fast LLMs for all items and invoking stronger models only for low‑confidence subsets) while retaining a single trainable decision surface.

## Acknowledgements

We thank contributors and users who reported issues and shared datasets. LabelFusion builds on the open‑source ecosystem, notably Hugging Face Transformers, scikit‑learn, PyTorch, and LLM provider SDKs.

## References
