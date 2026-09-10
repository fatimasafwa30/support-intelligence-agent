"""TF-IDF and Logistic Regression Intent Classifier for Customer Support.

Provides a reproducible, production-ready machine learning baseline for classifying
inbound customer support queries into canonical taxonomy intents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intents.silver_labeler import CANONICAL_INTENTS, normalize_text

ALLOWED_TRAINING_INTENTS: set[str] = {
    intent for intent in CANONICAL_INTENTS if intent != "other_unclear"
}


@dataclass(frozen=True)
class PredictionResult:
    """Structured result of intent classification prediction."""

    intent: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction result to dictionary."""
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


def default_preprocessor(text: str) -> str:
    """Normalize input text for vectorization."""
    return normalize_text(text)


class IntentClassifier:
    """Baseline Intent Classifier combining TF-IDF vectorization and Logistic Regression.

    Trained on high-confidence silver data to predict the 14 specific customer support intents.
    """

    def __init__(
        self,
        max_features: int | None = 15000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 2,
        C: float = 1.0,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
        max_iter: int = 1000,
        solver: str = "lbfgs",
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.solver = solver

        self.pipeline: Pipeline | None = None
        self.classes_: list[str] = []
        self._is_fitted: bool = False

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> IntentClassifier:
        """Fit the TF-IDF + Logistic Regression pipeline on labeled training texts.

        Args:
            texts: Sequence of training text strings.
            labels: Sequence of canonical intent labels.

        Returns:
            self
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"Mismatch between number of texts ({len(texts)}) and labels ({len(labels)})"
            )
        if len(texts) == 0:
            raise ValueError("Cannot train intent classifier on empty dataset.")

        unique_labels = set(labels)
        if "other_unclear" in unique_labels:
            raise ValueError(
                "Training on 'other_unclear' is not supported. Only specific intents are modeled."
            )

        invalid_labels = unique_labels - ALLOWED_TRAINING_INTENTS
        if invalid_labels:
            raise ValueError(
                f"Encountered labels outside canonical taxonomy: {sorted(invalid_labels)}"
            )

        # For very small test fixtures, min_df cannot exceed sample count
        effective_min_df = min(self.min_df, max(1, len(texts) // 2))

        vectorizer = TfidfVectorizer(
            preprocessor=default_preprocessor,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            min_df=effective_min_df,
        )

        clf = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            solver=self.solver,
        )

        self.pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf),
        ])

        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        self._is_fitted = True

        return self

    def predict_one(self, text: str) -> PredictionResult:
        """Predict intent and probability distribution for a single input text.

        Handles empty/whitespace strings safely.

        Args:
            text: Customer message text.

        Returns:
            PredictionResult containing intent, confidence, and class probabilities.
        """
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("IntentClassifier is not fitted. Call fit() or load() first.")

        safe_text = str(text or "")
        # Predict probabilities (shape: 1 x num_classes)
        proba_matrix = self.pipeline.predict_proba([safe_text])[0]

        best_idx = int(np.argmax(proba_matrix))
        best_intent = str(self.classes_[best_idx])
        best_confidence = float(proba_matrix[best_idx])

        # Map classes to float probabilities
        prob_dict = {
            cls_name: float(proba_matrix[idx])
            for idx, cls_name in enumerate(self.classes_)
        }

        return PredictionResult(
            intent=best_intent,
            confidence=best_confidence,
            probabilities=prob_dict,
        )

    def predict(self, texts: Sequence[str]) -> list[PredictionResult]:
        """Predict intents and probability distributions for a batch of messages."""
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("IntentClassifier is not fitted. Call fit() or load() first.")

        return [self.predict_one(t) for t in texts]

    def get_config(self) -> dict[str, Any]:
        """Return hyperparameter configuration dictionary."""
        return {
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "sublinear_tf": self.sublinear_tf,
            "min_df": self.min_df,
            "C": self.C,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "solver": self.solver,
            "classes": list(self.classes_),
            "num_classes": len(self.classes_),
        }

    def save(self, path: str | Path) -> Path:
        """Serialize the fitted pipeline and metadata to disk using joblib.

        Args:
            path: Destination file path.

        Returns:
            Path to saved artifact.
        """
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("Cannot save unfitted IntentClassifier.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": self.get_config(),
            "classes": self.classes_,
            "pipeline": self.pipeline,
        }
        joblib.dump(payload, save_path)
        return save_path

    @classmethod
    def load(cls, path: str | Path) -> IntentClassifier:
        """Load a serialized IntentClassifier from disk.

        Args:
            path: Path to serialized artifact.

        Returns:
            Restored IntentClassifier instance.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {load_path}")

        payload = joblib.load(load_path)
        config = payload.get("config", {})

        classifier = cls(
            max_features=config.get("max_features"),
            ngram_range=tuple(config.get("ngram_range", [1, 2])),
            sublinear_tf=config.get("sublinear_tf", True),
            min_df=config.get("min_df", 2),
            C=config.get("C", 1.0),
            class_weight=config.get("class_weight", "balanced"),
            random_state=config.get("random_state", 42),
            max_iter=config.get("max_iter", 1000),
            solver=config.get("solver", "lbfgs"),
        )
        classifier.classes_ = list(payload.get("classes", []))
        classifier.pipeline = payload.get("pipeline")
        classifier._is_fitted = True

        return classifier
