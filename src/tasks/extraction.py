"""
A2: Obligation Extraction Task

Extract obligations, rights, and prohibitions from treaty provisions.
Returns JSON list with type, actor, and action for each item.
"""

import json
import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate, system_message
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

SYSTEM_PROMPT = """You are an expert in international law and treaty analysis.
Your task is to extract all obligations, rights, and prohibitions from treaty text.

For each item, identify:
1. Type: "obligation", "right", or "prohibition"
2. Actor: Who bears this obligation/right (e.g., "State Party", "Member State", "Contracting Party")
3. Action: What they must/may/must not do

Respond in JSON format as a list of objects. Example:
[
  {"type": "obligation", "actor": "State Party", "action": "submit annual reports"},
  {"type": "prohibition", "actor": "Member State", "action": "transfer nuclear materials"}
]

Respond with ONLY the JSON array, no additional text."""

DATA_DIR = Path(__file__).parent.parent / "data"


def parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from response text."""
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return []


def semantic_similarity(text1: str, text2: str) -> float:
    """Compute token overlap similarity (Jaccard) between two strings."""
    # Tokenize and normalize
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    # Remove common stopwords for better matching
    stopwords = {'the', 'a', 'an', 'to', 'of', 'and', 'or', 'in', 'on', 'for', 'with', 'any', 'all'}
    tokens1 = tokens1 - stopwords
    tokens2 = tokens2 - stopwords

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union  # Jaccard similarity


def find_best_match(pred_action: str, gold_actions: list[str], threshold: float = 0.4) -> tuple[bool, float]:
    """Find if pred_action matches any gold action above threshold.

    Returns (matched, best_score).
    """
    best_score = 0.0
    for gold in gold_actions:
        score = semantic_similarity(pred_action, gold)
        best_score = max(best_score, score)
        if score >= threshold:
            return True, score
    return False, best_score


def compute_semantic_f1(predicted: list[dict], gold: list[dict]) -> dict[str, float]:
    """Compute F1 score with semantic (Jaccard) matching."""
    # Extract actions from predictions and gold
    pred_items = []
    for p in predicted:
        if isinstance(p, dict) and 'action' in p:
            pred_items.append({
                'action': p.get('action', ''),
                'type': p.get('type', ''),
                'actor': p.get('actor', '')
            })

    gold_items = []
    for g in gold:
        if isinstance(g, dict) and 'action' in g:
            gold_items.append({
                'action': g.get('action', ''),
                'type': g.get('type', ''),
                'actor': g.get('actor', '')
            })

    gold_actions = [g['action'] for g in gold_items]

    # Count true positives (predictions that match a gold)
    tp = 0
    matched_scores = []
    for p in pred_items:
        matched, score = find_best_match(p['action'], gold_actions)
        if matched:
            tp += 1
            matched_scores.append(score)

    # Calculate metrics
    precision = tp / max(len(pred_items), 1)
    recall = tp / max(len(gold_items), 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    avg_match_score = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_count": len(pred_items),
        "gold_count": len(gold_items),
        "true_positives": tp,
        "avg_match_score": avg_match_score,
    }


@scorer(metrics=[accuracy()])
def obligation_f1_scorer() -> Scorer:
    """Score obligation extraction using semantic F1 matching."""

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        predicted = parse_json_response(response)

        try:
            gold = json.loads(target.text)
            if not isinstance(gold, list):
                gold = [gold]
        except json.JSONDecodeError:
            gold = []

        # Use semantic F1 with Jaccard similarity matching
        metrics = compute_semantic_f1(predicted, gold)
        f1 = metrics["f1"]

        # Adjusted thresholds for semantic matching
        if f1 >= 0.6:
            value = CORRECT
        elif f1 >= 0.3:
            value = PARTIAL
        else:
            value = INCORRECT

        return Score(
            value=value,
            answer=response,
            explanation=f"F1={f1:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, TP={metrics['true_positives']}/{metrics['gold_count']}",
            metadata=metrics,
        )

    return score


@task
def obligation_extraction():
    """Extract obligations from treaty provisions."""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "extraction.jsonl")),
        solver=[
            system_message(SYSTEM_PROMPT),
            generate(),
        ],
        scorer=obligation_f1_scorer(),
    )
