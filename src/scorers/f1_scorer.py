"""
Custom F1 scorer for obligation extraction task.

Compares extracted obligations against gold standard using
action-based matching with fuzzy overlap.
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


def normalize_action(action: str) -> str:
    """Normalize action text for comparison."""
    return action.lower().strip()[:50]  # First 50 chars, lowercased


def compute_f1(predicted: list[dict], gold: list[dict]) -> dict[str, float]:
    """Compute F1 score based on action overlap."""
    # Extract normalized actions
    pred_actions = set()
    for p in predicted:
        if isinstance(p, dict) and 'action' in p:
            pred_actions.add(normalize_action(p['action']))

    gold_actions = set()
    for g in gold:
        if isinstance(g, dict) and 'action' in g:
            gold_actions.add(normalize_action(g['action']))

    # Compute overlap
    overlap = len(pred_actions & gold_actions)

    precision = overlap / max(len(pred_actions), 1)
    recall = overlap / max(len(gold_actions), 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_count": len(pred_actions),
        "gold_count": len(gold_actions),
        "overlap": overlap,
    }


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
