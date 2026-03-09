# LabelFusion Package Requirements Analysis

## Summary
This document provides a comprehensive analysis of all packages used in the LabelFusion repository.

## Status

### ✅ Packages Already in requirements.txt

The following packages are already specified in `requirements.txt`:

#### Core ML/AI Libraries
- **torch** (2.7.1) - PyTorch deep learning framework
- **transformers** (4.55.0) - HuggingFace transformers
- **scikit-learn** (1.7.1) - ML utilities
- **numpy** (2.3.3) - Numerical computing
- **pandas** (2.3.3) - Data manipulation

#### LLM API Clients  
- **openai** (1.97.0) - OpenAI API client
- **anthropic** (0.58.2) - Anthropic Claude API client
- **google-generativeai** (0.8.5) - Google Gemini API client

#### Data Processing
- **datasets** (4.1.1) - HuggingFace datasets
- **tokenizers** (0.21.4) - Text tokenization

#### Utilities
- **tqdm** (4.67.1) - Progress bars
- **python-dotenv** (1.1.1) - Environment variable management
- **PyYAML** (6.0.2) - YAML parsing
- **requests** (2.32.4) - HTTP requests

#### Visualization
- **matplotlib** (3.10.6) - Plotting
- **seaborn** (0.13.2) - Statistical visualization

#### Development Tools
- **pytest** (8.4.1) - Testing framework
- **pytest-asyncio** (1.1.0) - Async testing
- **pytest-cov** (6.2.1) - Coverage
- **black** (25.1.0) - Code formatting
- **flake8** (7.3.0) - Linting
- **mypy** (1.17.0) - Type checking
- **pre_commit** (4.2.0) - Git hooks

#### Jupyter/Notebook Support
- **ipykernel** (6.29.5) - Jupyter kernel
- **ipython** (9.4.0) - Interactive Python
- **jupyter_client** (8.6.3) - Jupyter client
- **jupyter_core** (5.8.1) - Jupyter core

#### NVIDIA CUDA Libraries
- nvidia-cublas-cu12, nvidia-cuda-cupti-cu12, nvidia-cuda-nvrtc-cu12
- nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12
- nvidia-cufile-cu12, nvidia-curand-cu12, nvidia-cusolver-cu12
- nvidia-cusparse-cu12, nvidia-cusparselt-cu12, nvidia-nccl-cu12
- nvidia-nvjitlink-cu12, nvidia-nvtx-cu12

#### Other Dependencies
- **accelerate** (1.10.0) - Model acceleration
- **safetensors** (0.6.2) - Safe tensor serialization
- **scipy** (1.16.1) - Scientific computing
- And many more...

---

## ⚠️ MISSING PACKAGES

### Critical Missing Package
**beautifulsoup4** - Used in `reuters-overview.ipynb` for parsing SGML files

#### Usage Location:
- File: `reuters-overview.ipynb`
- Import: `from bs4 import BeautifulSoup`
- Purpose: Parsing Reuters-21578 SGML dataset files

### Recommended Package
**lxml** - Parser backend for BeautifulSoup (optional but recommended)
- Provides faster and more robust parsing
- Better handling of malformed HTML/SGML

---

## 📦 Required Action

### Add to requirements.txt:
```
beautifulsoup4==4.12.3
lxml==5.1.0
```

### Installation Command:
```bash
pip install beautifulsoup4 lxml
```

---

## Package Usage by Module

### textclassify/ml/
- torch, transformers, numpy, pandas
- Dataset, DataLoader from torch.utils.data
- RoBERTa models from transformers

### textclassify/llm/
- openai, anthropic, google-generativeai
- pandas, asyncio
- Custom prediction caching

### textclassify/ensemble/
- torch, numpy, pandas
- sklearn (train_test_split, IsotonicRegression, CalibratedClassifierCV)

### textclassify/utils/
- pandas, numpy
- yaml, pickle, json
- hashlib, pathlib

### tests/
- pytest, pandas
- datasets library for loading test data

### Notebooks
- pandas, matplotlib, seaborn
- datasets (HuggingFace)
- **beautifulsoup4 (MISSING)**
- numpy, re, pathlib

---

## Standard Library Modules Used
(No installation required - part of Python stdlib)

- os, sys, pathlib
- re, json, csv, pickle
- typing, dataclasses, abc, enum
- datetime, time
- asyncio, logging
- collections, itertools
- hashlib, warnings

---

## Development Environment

The project uses:
- Python 3.x (exact version not specified in files checked)
- Virtual environment recommended
- CUDA support for GPU acceleration (optional but recommended for ML models)

---

## Notes

1. The `-e git+https://github.com/ChrisW09/classifyfusion.git@...` entry in requirements.txt indicates the package itself is installed in editable mode from a git repository.

2. Many NVIDIA CUDA packages are included, suggesting GPU support is important for this project.

3. The project supports multiple LLM providers: OpenAI, Anthropic (Claude), Google (Gemini), and DeepSeek.

4. Testing infrastructure is comprehensive with pytest, async support, and coverage reporting.

---

## Recommendations

1. **Immediate**: Add `beautifulsoup4` and `lxml` to requirements.txt
2. Consider pinning Python version in a `.python-version` or documentation
3. Consider separating dev dependencies from runtime dependencies
4. Document GPU requirements separately if CUDA packages are optional
