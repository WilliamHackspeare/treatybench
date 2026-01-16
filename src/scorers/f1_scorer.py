"""
Custom F1 scorer for obligation extraction task.

Compares extracted obligations against gold standard using
semantic similarity matching (Sentence-BERT) with type/actor bonuses.

Improvements over simple token overlap:
- Semantic similarity captures paraphrases and synonymy
- Type matching bonus for obligation/right/prohibition
- Actor matching bonus for entity recognition
"""

import json
import re
from typing import Any

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    scorer,
    accuracy,
    CORRECT,
    INCORRECT,
    PARTIAL,
)
from inspect_ai.solver import TaskState

from .semantic_scorer import get_semantic_scorer, semantic_similarity


def parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from response text."""
    # Try direct parse
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in text
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return []


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def compute_obligation_similarity(pred: dict, gold: dict) -> float:
    """Compute similarity score between predicted and gold obligation.

    Combines:
    - Action semantic similarity (0-1)
    - Type match bonus (+0.1)
    - Actor match bonus (+0.1)

    Returns score in range [0, 1.2] but capped at 1.0
    """
    score = 0.0

    # Action similarity (primary component)
    pred_action = pred.get('action', '')
    gold_action = gold.get('action', '')

    if pred_action and gold_action:
        score = semantic_similarity(pred_action, gold_action)

    # Type match bonus
    pred_type = normalize_text(pred.get('type', ''))
    gold_type = normalize_text(gold.get('type', ''))
    if pred_type and gold_type and pred_type == gold_type:
        score += 0.1

    # Actor match bonus (semantic similarity for flexibility)
    pred_actor = pred.get('actor', '')
    gold_actor = gold.get('actor', '')
    if pred_actor and gold_actor:
        actor_sim = semantic_similarity(pred_actor, gold_actor)
        if actor_sim >= 0.7:  # High similarity threshold for actors
            score += 0.1

    return min(score, 1.0)


def compute_f1_semantic(
    predicted: list[dict],
    gold: list[dict],
    threshold: float = 0.6
) -> dict[str, float]:
    """Compute F1 score using semantic matching.

    Uses greedy matching: for each gold obligation, find best matching
    prediction above threshold. Matched predictions are consumed.

    Args:
        predicted: List of predicted obligation dicts
        gold: List of gold obligation dicts
        threshold: Minimum similarity for a match

    Returns:
        Dict with precision, recall, f1, and detailed metrics
    """
    if not gold:
        # No gold standards - if predictions exist, precision is 0
        return {
            "precision": 0.0 if predicted else 1.0,
            "recall": 1.0,  # Vacuously true
            "f1": 0.0 if predicted else 1.0,
            "pred_count": len(predicted),
            "gold_count": 0,
            "matches": 0,
            "avg_match_score": 0.0,
        }

    if not predicted:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "pred_count": 0,
            "gold_count": len(gold),
            "matches": 0,
            "avg_match_score": 0.0,
        }

    # Greedy matching
    available_preds = list(range(len(predicted)))
    matches = 0
    match_scores = []

    for g in gold:
        best_score = 0.0
        best_pred_idx = -1

        for pred_idx in available_preds:
            sim = compute_obligation_similarity(predicted[pred_idx], g)
            if sim > best_score:
                best_score = sim
                best_pred_idx = pred_idx

        if best_score >= threshold and best_pred_idx >= 0:
            matches += 1
            match_scores.append(best_score)
            available_preds.remove(best_pred_idx)

    precision = matches / len(predicted)
    recall = matches / len(gold)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_count": len(predicted),
        "gold_count": len(gold),
        "matches": matches,
        "avg_match_score": avg_match_score,
    }


# Legacy function for backward compatibility
def compute_f1(predicted: list[dict], gold: list[dict]) -> dict[str, float]:
    """Compute F1 score - now uses semantic matching."""
    return compute_f1_semantic(predicted, gold)


@scorer(metrics=[accuracy()])
def obligation_f1_scorer() -> Scorer:
    """Score obligation extraction using F1."""

    async def score(state: TaskState, target: Target) -> Score:
        # Get model response
        response = state.output.completion

        # Parse predicted obligations
        predicted = parse_json_response(response)

        # Parse gold standard (target is JSON string)
        try:
            gold = json.loads(target.text)
            if not isinstance(gold, list):
                gold = [gold]
        except json.JSONDecodeError:
            gold = []

        # Compute F1
        metrics = compute_f1(predicted, gold)

        # Determine score value
        f1 = metrics["f1"]
        if f1 >= 0.8:
            value = CORRECT
        elif f1 >= 0.4:
            value = PARTIAL
        else:
            value = INCORRECT

        return Score(
            value=value,
            answer=response,
            explanation=f"F1={f1:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}",
            metadata=metrics,
        )

    return score
