"""TF-IDF Historical-Resolution Retriever for Customer Support.

Indexes historical customer-brand resolution pairs and retrieves the most relevant
past resolutions for an incoming customer problem statement.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intents.silver_labeler import normalize_text
from src.retrieval.historical_resolution import HistoricalResolution, RetrievalResult

VALID_INDEX_FIELDS = {"customer", "both", "brand"}


def default_preprocessor(text: str) -> str:
    """Normalize input text for retrieval vectorization."""
    return normalize_text(text)


class TFIDFRetriever:
    """TF-IDF Historical-Resolution Retriever.

    Matches an incoming customer query against historical support resolutions using
    sparse TF-IDF vector representations and cosine similarity.
    """

    def __init__(
        self,
        index_field: str = "customer",
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = 50000,
        sublinear_tf: bool = True,
        min_df: int = 2,
    ) -> None:
        """Initialize retriever configuration.

        Args:
            index_field: The field(s) used for matching:
                         - 'customer': Match query against past customer messages (DEFAULT).
                         - 'both': Match query against concatenated customer and brand texts.
                         - 'brand': Match query against brand response texts.
            ngram_range: Lower and upper boundary of range of n-values for n-grams.
            max_features: Maximum vocabulary size.
            sublinear_tf: Apply sublinear scaling (1 + log(tf)).
            min_df: Minimum document frequency for terms.
        """
        if index_field not in VALID_INDEX_FIELDS:
            raise ValueError(
                f"Invalid index_field: {index_field}. Must be one of {sorted(VALID_INDEX_FIELDS)}"
            )

        self.index_field = index_field
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df

        self.vectorizer: TfidfVectorizer | None = None
        self.doc_matrix: sparse.csr_matrix | None = None
        self.resolutions: list[HistoricalResolution] = []
        self._is_fitted: bool = False

    def _extract_index_text(self, resolution: HistoricalResolution) -> str:
        """Extract the text to index for a single historical resolution."""
        if self.index_field == "customer":
            return resolution.customer_text
        if self.index_field == "both":
            return f"{resolution.customer_text} {resolution.brand_text}".strip()
        if self.index_field == "brand":
            return resolution.brand_text
        raise ValueError(f"Unsupported index_field: {self.index_field}")

    def fit(self, resolutions: Sequence[HistoricalResolution]) -> TFIDFRetriever:
        """Fit the TF-IDF vectorizer and index all historical resolutions.

        Args:
            resolutions: Sequence of historical resolution objects.

        Returns:
            self
        """
        if not resolutions:
            raise ValueError("Cannot fit TFIDFRetriever on an empty resolution sequence.")

        self.resolutions = list(resolutions)
        corpus_texts = [self._extract_index_text(r) for r in self.resolutions]

        effective_min_df = min(self.min_df, max(1, len(corpus_texts) // 2))

        self.vectorizer = TfidfVectorizer(
            preprocessor=default_preprocessor,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            min_df=effective_min_df,
            norm="l2",
        )

        self.doc_matrix = self.vectorizer.fit_transform(corpus_texts)
        self._is_fitted = True
        return self

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_intent: str | None = None,
        exclude_conversation_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-K most similar historical resolutions for a query.

        Args:
            query: The incoming customer query text.
            top_k: Number of candidate resolutions to return.
            min_score: Minimum cosine similarity score threshold [0.0, 1.0].
            filter_intent: Optional canonical intent to restrict candidates.
            exclude_conversation_id: Optional conversation ID to strictly exclude
                from retrieval candidates (prevents query-conversation self-leakage).

        Returns:
            List of RetrievalResult objects sorted descending by similarity score.
        """
        if not self._is_fitted or self.vectorizer is None or self.doc_matrix is None:
            raise RuntimeError("TFIDFRetriever is not fitted. Call fit() or load() first.")

        if top_k <= 0:
            return []

        safe_query = str(query or "").strip()
        if not safe_query:
            return []

        # Transform query into unit-norm sparse vector (1 x V)
        query_vec = self.vectorizer.transform([safe_query])

        # If query contains no known terms, dot product is all zeros
        if query_vec.nnz == 0:
            return []

        # Cosine similarity via dot product: (N x V) . (V x 1) -> (N, 1)
        raw_scores = self.doc_matrix.dot(query_vec.T).toarray().ravel()

        # Build candidate mask
        mask = raw_scores >= min_score

        if exclude_conversation_id is not None:
            for idx, res in enumerate(self.resolutions):
                if res.conversation_id == exclude_conversation_id:
                    mask[idx] = False

        if filter_intent is not None:
            for idx, res in enumerate(self.resolutions):
                if res.intent != filter_intent:
                    mask[idx] = False

        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_scores = raw_scores[valid_indices]

        # Top-K sorting
        if len(valid_scores) > top_k:
            partition_idx = np.argpartition(-valid_scores, top_k)[:top_k]
            top_sub_indices = partition_idx[np.argsort(-valid_scores[partition_idx])]
        else:
            top_sub_indices = np.argsort(-valid_scores)

        selected_global_indices = valid_indices[top_sub_indices]

        results: list[RetrievalResult] = []
        for rank, g_idx in enumerate(selected_global_indices, start=1):
            results.append(
                RetrievalResult(
                    resolution=self.resolutions[g_idx],
                    score=float(raw_scores[g_idx]),
                    rank=rank,
                )
            )

        return results

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: int = 5,
        min_score: float = 0.0,
        filter_intents: Sequence[str | None] | None = None,
        exclude_conversation_ids: Sequence[str | None] | None = None,
    ) -> list[list[RetrievalResult]]:
        """Retrieve top-K candidates for a batch of queries."""
        n_queries = len(queries)
        f_intents = filter_intents or [None] * n_queries
        e_cids = exclude_conversation_ids or [None] * n_queries

        results: list[list[RetrievalResult]] = []
        for i, q in enumerate(queries):
            results.append(
                self.retrieve(
                    query=q,
                    top_k=top_k,
                    min_score=min_score,
                    filter_intent=f_intents[i],
                    exclude_conversation_id=e_cids[i],
                )
            )
        return results

    def get_config(self) -> dict[str, Any]:
        """Return hyperparameter configuration dictionary."""
        return {
            "index_field": self.index_field,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "min_df": self.min_df,
            "num_indexed_resolutions": len(self.resolutions),
        }

    def save(self, path: str | Path) -> Path:
        """Serialize retriever state to disk using joblib.

        Args:
            path: Destination file path.

        Returns:
            Path to saved artifact.
        """
        if not self._is_fitted or self.vectorizer is None or self.doc_matrix is None:
            raise RuntimeError("Cannot save unfitted TFIDFRetriever.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": self.get_config(),
            "vectorizer": self.vectorizer,
            "doc_matrix": self.doc_matrix,
            "resolutions": [r.to_dict() for r in self.resolutions],
        }
        joblib.dump(payload, save_path)
        return save_path

    @classmethod
    def load(cls, path: str | Path) -> TFIDFRetriever:
        """Load a serialized TFIDFRetriever from disk.

        Args:
            path: Path to serialized artifact.

        Returns:
            Restored TFIDFRetriever instance.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Retriever artifact not found at: {load_path}")

        payload = joblib.load(load_path)
        config = payload.get("config", {})

        retriever = cls(
            index_field=config.get("index_field", "customer"),
            ngram_range=tuple(config.get("ngram_range", [1, 2])),
            max_features=config.get("max_features", 50000),
            sublinear_tf=config.get("sublinear_tf", True),
            min_df=config.get("min_df", 2),
        )

        retriever.vectorizer = payload.get("vectorizer")
        retriever.doc_matrix = payload.get("doc_matrix")
        raw_res = payload.get("resolutions", [])
        retriever.resolutions = [HistoricalResolution.from_dict(r) for r in raw_res]
        retriever._is_fitted = True

        return retriever
