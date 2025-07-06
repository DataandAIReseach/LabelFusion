"""Setup script for textclassify package."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="textclassify",
    version="0.1.0",
    author="TextClassify Team",
    author_email="contact@textclassify.com",
    description="A comprehensive Python package for multi-class and multi-label text classification using LLMs and traditional ML models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/textclassify",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "ml": [
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "scikit-learn>=1.1.0",
        ],
        "config": [
            "pyyaml>=6.0",
        ],
        "all": [
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "scikit-learn>=1.1.0",
            "pyyaml>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "textclassify=textclassify.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "textclassify": [
            "examples/*.py",
            "config/*.yaml",
        ],
    },
    keywords=[
        "text classification",
        "machine learning",
        "natural language processing",
        "llm",
        "ensemble methods",
        "multi-class",
        "multi-label",
        "openai",
        "claude",
        "gemini",
        "roberta",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/textclassify/issues",
        "Source": "https://github.com/your-org/textclassify",
        "Documentation": "https://textclassify.readthedocs.io",
    },
)

