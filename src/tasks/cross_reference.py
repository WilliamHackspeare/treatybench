"""
A4: Cross-Reference Resolution Task

Given a treaty provision with cross-references to other articles or definitions,
identify and extract what the reference points to.
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
Your task is to resolve cross-references in treaty provisions.

You will be given:
1. A provision containing a cross-reference (e.g., "as defined in Article II", "pursuant to paragraph 3")
2. The relevant context from the treaty containing the referenced section

Your task is to:
1. Identify what is being referenced
2. Extract the exact text or definition being referred to
3. Explain how the reference applies

Respond in JSON format:
{
  "reference_type": "definition|obligation|procedure|scope",
  "referenced_text": "The exact text from the treaty that is being referenced",
  "summary": "Brief summary of what the reference establishes"
}

Respond with ONLY the JSON object, no additional text."""

DATA_DIR = Path(__file__).parent.parent / "data"


def parse_json_response(text: str) -> dict:
    """Extract JSON object from response text."""
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
        return {}
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {}


def text_similarity(text1: str, text2: str) -> float:
    """Compute token overlap similarity between two strings."""
    if not text1 or not text2:
        return 0.0

    # Normalize
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'to', 'of', 'and', 'or', 'in', 'on', 'for', 'with', 'any', 'all', 'is', 'are', 'be', 'been', 'being'}
    tokens1 = tokens1 - stopwords
    tokens2 = tokens2 - stopwords

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union


def score_cross_reference(predicted: dict, gold: dict) -> dict:
    """Score cross-reference resolution."""
    # Extract referenced text
    pred_text = predicted.get("referenced_text", "")
    gold_text = gold.get("referenced_text", "")

    # Primary metric: text similarity
    text_sim = text_similarity(pred_text, gold_text)

    # Bonus for correct reference type
    pred_type = predicted.get("reference_type", "").lower()
    gold_type = gold.get("reference_type", "").lower()
    type_match = 1.0 if pred_type == gold_type else 0.0

    # Combined score (80% text, 20% type)
    combined_score = 0.8 * text_sim + 0.2 * type_match

    return {
        "text_similarity": text_sim,
        "type_match": type_match,
        "combined_score": combined_score,
    }


@scorer(metrics=[accuracy()])
def cross_reference_scorer() -> Scorer:
    """Score cross-reference resolution."""

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        predicted = parse_json_response(response)

        try:
            gold = json.loads(target.text)
            if not isinstance(gold, dict):
                gold = {}
        except json.JSONDecodeError:
            gold = {}

        metrics = score_cross_reference(predicted, gold)
        combined = metrics["combined_score"]

        if combined >= 0.6:
            value = CORRECT
        elif combined >= 0.3:
            value = PARTIAL
        else:
            value = INCORRECT

        return Score(
            value=value,
            answer=response,
            explanation=f"Combined={combined:.3f}, TextSim={metrics['text_similarity']:.3f}, TypeMatch={metrics['type_match']:.0f}",
            metadata=metrics,
        )

    return score


@task
def cross_reference_resolution():
    """Resolve cross-references in treaty provisions (A4)."""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "cross_reference.jsonl")),
        solver=[
            system_message(SYSTEM_PROMPT),
            generate(),
        ],
        scorer=cross_reference_scorer(),
    )
