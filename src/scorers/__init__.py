"""
TreatyBench Scoring Methods

Improved scorers addressing LLM-as-judge biases:
- f1_scorer: Obligation extraction F1 scoring
- semantic_scorer: Sentence-BERT similarity (replaces Jaccard)
- multi_judge: Multi-model judging with explanation-first prompting
"""

from .f1_scorer import obligation_f1_scorer

from .semantic_scorer import (
    SemanticSimilarityScorer,
    get_semantic_scorer,
    semantic_similarity,
    find_best_semantic_match,
)

from .multi_judge import (
    multi_judge_scorer,
    multi_model_judge,
    get_multi_judge_scorer,
    DEFAULT_JUDGE_MODELS,
)

__all__ = [
    # F1 scoring
    "obligation_f1_scorer",
    # Semantic similarity
    "SemanticSimilarityScorer",
    "get_semantic_scorer",
    "semantic_similarity",
    "find_best_semantic_match",
    # Multi-model judging
    "multi_judge_scorer",
    "multi_model_judge",
    "get_multi_judge_scorer",
    "DEFAULT_JUDGE_MODELS",
]
