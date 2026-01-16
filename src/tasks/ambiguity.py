"""
A3: Ambiguity Detection Task

Identify ambiguous or vague terms in treaty provisions that could
lead to interpretation disputes or enforcement difficulties.
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

SYSTEM_PROMPT = """You are an expert in international law and treaty interpretation.
Your task is to identify ambiguous or vague terms in treaty provisions that could
lead to interpretation disputes, enforcement difficulties, or legal uncertainty.

For each provision, identify:
1. Ambiguous terms or phrases (words/phrases that lack clear definition or criteria)
2. For each term, explain WHY it is ambiguous

Common sources of ambiguity in treaties include:
- Undefined qualitative terms ("appropriate", "reasonable", "adequate")
- Vague temporal references ("promptly", "without delay", "as soon as possible")
- Unclear scope ("significant", "substantial", "material")
- Discretionary language ("may", "should", "endeavor to")
- Missing criteria or thresholds

Respond in JSON format:
{
  "ambiguous_terms": [
    {"term": "...", "reason": "..."},
    {"term": "...", "reason": "..."}
  ]
}

If the provision is unusually clear with no significant ambiguities, respond:
{"ambiguous_terms": []}

Respond with ONLY the JSON object, no additional text."""

DATA_DIR = Path(__file__).parent.parent / "data"


def parse_json_response(text: str) -> dict:
    """Extract JSON object from response text."""
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
        return {"ambiguous_terms": []}
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {"ambiguous_terms": []}


def normalize_term(term: str) -> str:
    """Normalize a term for comparison."""
    return term.lower().strip().strip('"\'')


def term_overlap(pred_term: str, gold_term: str) -> bool:
    """Check if predicted term overlaps with gold term."""
    pred_norm = normalize_term(pred_term)
    gold_norm = normalize_term(gold_term)

    # Exact match
    if pred_norm == gold_norm:
        return True

    # Substring match (either direction)
    if pred_norm in gold_norm or gold_norm in pred_norm:
        return True

    # Word overlap
    pred_words = set(pred_norm.split())
    gold_words = set(gold_norm.split())
    if pred_words & gold_words:
        overlap = len(pred_words & gold_words) / max(len(gold_words), 1)
        if overlap >= 0.5:
            return True

    return False


def compute_ambiguity_f1(predicted: dict, gold: dict) -> dict:
    """Compute F1 score for ambiguity detection."""
    pred_terms = predicted.get("ambiguous_terms", [])
    gold_terms = gold.get("ambiguous_terms", [])

    if not isinstance(pred_terms, list):
        pred_terms = []
    if not isinstance(gold_terms, list):
        gold_terms = []

    # Extract term strings
    pred_term_strs = []
    for t in pred_terms:
        if isinstance(t, dict) and "term" in t:
            pred_term_strs.append(t["term"])
        elif isinstance(t, str):
            pred_term_strs.append(t)

    gold_term_strs = []
    for t in gold_terms:
        if isinstance(t, dict) and "term" in t:
            gold_term_strs.append(t["term"])
        elif isinstance(t, str):
            gold_term_strs.append(t)

    # Count true positives
    tp = 0
    matched_gold = set()
    for pred in pred_term_strs:
        for i, gold in enumerate(gold_term_strs):
            if i not in matched_gold and term_overlap(pred, gold):
                tp += 1
                matched_gold.add(i)
                break

    precision = tp / max(len(pred_term_strs), 1)
    recall = tp / max(len(gold_term_strs), 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_count": len(pred_term_strs),
        "gold_count": len(gold_term_strs),
        "true_positives": tp,
    }


@scorer(metrics=[accuracy()])
def ambiguity_f1_scorer() -> Scorer:
    """Score ambiguity detection using F1 matching."""

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        predicted = parse_json_response(response)

        try:
            gold = json.loads(target.text)
            if not isinstance(gold, dict):
                gold = {"ambiguous_terms": []}
        except json.JSONDecodeError:
            gold = {"ambiguous_terms": []}

        metrics = compute_ambiguity_f1(predicted, gold)
        f1 = metrics["f1"]

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
def ambiguity_detection():
    """Detect ambiguous terms in treaty provisions (A3)."""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "ambiguity.jsonl")),
        solver=[
            system_message(SYSTEM_PROMPT),
            generate(),
        ],
        scorer=ambiguity_f1_scorer(),
    )
