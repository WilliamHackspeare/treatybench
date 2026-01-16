"""
Semantic Similarity Scorer using Sentence-BERT

Provides improved semantic matching for extraction tasks using
sentence embeddings instead of simple token overlap.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Any, TYPE_CHECKING

# Try to import sentence-transformers, fall back to Jaccard if unavailable
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    SentenceTransformer = Any  # Type stub for when not installed
    print("Warning: sentence-transformers not installed. Using Jaccard similarity fallback.")


class SemanticSimilarityScorer:
    """Compute semantic similarity using Sentence-BERT or fallback to Jaccard."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> Optional[SentenceTransformer]:
        """Lazy load the model to avoid loading at import time."""
        if SBERT_AVAILABLE and self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Fallback Jaccard similarity using token overlap."""
        if not text1 or not text2:
            return 0.0

        # Tokenize and normalize
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'to', 'of', 'and', 'or', 'in', 'on', 'for',
            'with', 'any', 'all', 'is', 'are', 'be', 'been', 'being',
            'shall', 'may', 'will', 'would', 'should', 'could', 'must'
        }
        tokens1 = tokens1 - stopwords
        tokens2 = tokens2 - stopwords

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Uses Sentence-BERT if available, otherwise falls back to Jaccard.
        """
        if not text1 or not text2:
            return 0.0

        if SBERT_AVAILABLE and self.model is not None:
            try:
                embeddings = self.model.encode([text1, text2])
                return self.cosine_similarity(embeddings[0], embeddings[1])
            except Exception:
                # Fall back to Jaccard on any error
                return self.jaccard_similarity(text1, text2)
        else:
            return self.jaccard_similarity(text1, text2)


# Global instance for reuse
_scorer_instance: Optional[SemanticSimilarityScorer] = None


def get_semantic_scorer() -> SemanticSimilarityScorer:
    """Get or create a global semantic scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = SemanticSimilarityScorer()
    return _scorer_instance


def semantic_similarity(text1: str, text2: str) -> float:
    """Convenience function for computing semantic similarity."""
    return get_semantic_scorer().similarity(text1, text2)


def find_best_semantic_match(
    pred_text: str,
    gold_texts: list[str],
    threshold: float = 0.5
) -> tuple[bool, float, int]:
    """
    Find the best matching gold text for a prediction.

    Returns:
        (matched, best_score, best_index)
    """
    scorer = get_semantic_scorer()
    best_score = 0.0
    best_idx = -1

    for i, gold in enumerate(gold_texts):
        score = scorer.similarity(pred_text, gold)
        if score > best_score:
            best_score = score
            best_idx = i
        if score >= threshold:
            return True, score, i

    return best_score >= threshold, best_score, best_idx
