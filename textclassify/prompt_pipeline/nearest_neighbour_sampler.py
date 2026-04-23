"""
NearestNeighbourSampler: Selects the most semantically similar training examples
for each test text using sentence embeddings and nearest neighbour search.

Replaces random sampling in fill_train_data_prompt with semantically relevant
few-shot examples — improving prompt quality for LLM classification.
"""

import numpy as np
import pandas as pd
from typing import Optional


class NearestNeighbourSampler:
    """
    Builds an embedding table of all training texts and retrieves the
    k most semantically similar examples for a given query text.

    Usage:
        sampler = NearestNeighbourSampler(model_name="all-MiniLM-L6-v2")
        sampler.fit(train_df, text_column="text")
        similar_df = sampler.sample(query_text="some test text", k=5)

    For multilingual datasets use:
        NearestNeighbourSampler(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: SentenceTransformer model name.
                        'all-MiniLM-L6-v2' is fast and good for English.
                        'paraphrase-multilingual-MiniLM-L12-v2' for multilingual.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for NearestNeighbourSampler. "
                "Install it with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._train_df: Optional[pd.DataFrame] = None
        self._embeddings: Optional[np.ndarray] = None
        self._text_column: Optional[str] = None
        self._is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame, text_column: str) -> "NearestNeighbourSampler":
        """
        Compute and store embeddings for all texts in the training DataFrame.
        Should be called once before any sample() calls.

        Args:
            train_df: DataFrame containing training examples
            text_column: Name of the column containing text

        Returns:
            self (for method chaining)
        """
        if train_df is None or train_df.empty:
            raise ValueError("train_df must be a non-empty DataFrame")
        if text_column not in train_df.columns:
            raise ValueError(f"Column '{text_column}' not found in train_df")

        self._train_df = train_df.reset_index(drop=True)
        self._text_column = text_column

        print(f"NearestNeighbourSampler: computing embeddings for {len(train_df)} training texts...")
        texts = self._train_df[text_column].tolist()
        self._embeddings = self._model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # enables cosine similarity via dot product
        )
        self._is_fitted = True
        print(f"NearestNeighbourSampler: embedding table ready ({self._embeddings.shape})")

        return self

    def sample(self, query_text: str, k: int = 5) -> pd.DataFrame:
        """
        Return the k most semantically similar training examples for query_text.

        Args:
            query_text: The test text to find similar training examples for
            k: Number of nearest neighbours to return

        Returns:
            DataFrame with k most similar training rows, ordered by similarity
            (most similar first)
        """
        if not self._is_fitted:
            raise RuntimeError(
                "NearestNeighbourSampler is not fitted yet. Call fit() first."
            )

        # Encode query text
        query_embedding = self._model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Cosine similarity via dot product (embeddings are L2-normalized)
        similarities = self._embeddings @ query_embedding

        # Get top-k indices sorted by descending similarity
        k = min(k, len(self._train_df))
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

        return self._train_df.iloc[top_k_indices].reset_index(drop=True)

    @property
    def is_fitted(self) -> bool:
        """True if the embedding table has been computed."""
        return self._is_fitted

    @property
    def n_train_samples(self) -> int:
        """Number of training samples in the embedding table."""
        return len(self._train_df) if self._train_df is not None else 0