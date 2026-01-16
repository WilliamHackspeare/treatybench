"""
Multi-Model Judge Scorer

Addresses LLM-as-judge biases by:
1. Using multiple judge models and averaging scores
2. Requiring explanation before scoring (explanation-first)
3. Position swapping for pairwise comparisons (future)

Based on recommendations from:
- Zheng et al. (2024): LLM-as-Judge survey
- Wang et al. (2024): CALM framework for bias quantification
"""

import re
import asyncio
from typing import Optional
from statistics import mean, stdev

from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, CORRECT, INCORRECT, PARTIAL
from inspect_ai.solver import TaskState
from inspect_ai.model import get_model, Model, GenerateConfig


# Primary judge model - using Opus 4.5 for highest quality judgments
DEFAULT_JUDGE_MODELS = [
    "anthropic/claude-opus-4-5-20251101",
]

# Cross-validation models for self-enhancement bias analysis
# Used to re-score Opus outputs for Appendix comparison
CROSS_VALIDATION_JUDGES = [
    "openai/gpt-5.2",
    "google/gemini-3.0-pro",
]

# All available judges for multi-model scoring
ALL_JUDGE_MODELS = [
    "anthropic/claude-opus-4-5-20251101",
    "openai/gpt-5.2",
    "google/gemini-3.0-pro",
]

# Explanation-first template improves alignment (Chiang & Lee 2023)
EXPLANATION_FIRST_TEMPLATE = """You are evaluating the quality of a generated treaty provision.

TASK SPECIFICATION:
{question}

GENERATED RESPONSE:
{answer}

REFERENCE (for context, not exact match required):
{target}

First, analyze the response by considering:
1. Legal Soundness: Is the language legally valid and enforceable?
2. Specificity: Are terms and obligations clearly defined?
3. Implementability: Could this be practically implemented?
4. Completeness: Does it address all aspects of the specification?
5. Treaty Conventions: Does it follow standard treaty drafting conventions?

After your analysis, provide a final rating on a scale of 1-5:
- 5: Excellent - Could serve as viable treaty language
- 4: Good - Addresses requirements with minor gaps
- 3: Adequate - Meets basic requirements but lacks specificity
- 2: Poor - Misses key requirements or contains significant issues
- 1: Failing - Incoherent, legally invalid, or misses the specification

Format your response as:
ANALYSIS: [Your detailed analysis]
GRADE: [1-5]
"""


async def get_judge_score(
    model: Model,
    question: str,
    answer: str,
    target: str,
    template: str = EXPLANATION_FIRST_TEMPLATE,
) -> tuple[Optional[float], str]:
    """Get a score from a single judge model.

    Returns:
        (score, explanation) tuple. Score is None if parsing failed.
    """
    prompt = template.format(question=question, answer=answer, target=target)

    try:
        response = await model.generate(
            prompt,
            config=GenerateConfig(temperature=0.0, max_tokens=1000),
        )

        text = response.completion

        # Extract grade
        match = re.search(r"GRADE:\s*([1-5])", text)
        if match:
            score = float(match.group(1))
            return score, text
        else:
            return None, text

    except Exception as e:
        return None, f"Error: {str(e)}"


async def multi_model_judge(
    question: str,
    answer: str,
    target: str,
    judge_models: list[str] = DEFAULT_JUDGE_MODELS,
    template: str = EXPLANATION_FIRST_TEMPLATE,
) -> dict:
    """Get scores from multiple judge models and aggregate.

    Returns:
        dict with keys: mean_score, scores, std_dev, explanations, agreement
    """
    # Load models
    models = [get_model(model_name) for model_name in judge_models]

    # Get scores concurrently
    tasks = [
        get_judge_score(model, question, answer, target, template)
        for model in models
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    scores = []
    explanations = {}

    for i, result in enumerate(results):
        model_name = judge_models[i]
        if isinstance(result, Exception):
            explanations[model_name] = f"Error: {str(result)}"
        else:
            score, explanation = result
            explanations[model_name] = explanation
            if score is not None:
                scores.append(score)

    if not scores:
        return {
            "mean_score": None,
            "scores": [],
            "std_dev": None,
            "explanations": explanations,
            "agreement": 0.0,
        }

    mean_score = mean(scores)
    std = stdev(scores) if len(scores) > 1 else 0.0

    # Agreement: proportion of scores within 1 point of the mean
    agreement = sum(1 for s in scores if abs(s - mean_score) <= 1) / len(scores)

    return {
        "mean_score": mean_score,
        "scores": scores,
        "std_dev": std,
        "explanations": explanations,
        "agreement": agreement,
    }


@scorer(metrics=[accuracy()])
def multi_judge_scorer(
    judge_models: list[str] = DEFAULT_JUDGE_MODELS,
    threshold_excellent: float = 4.0,
    threshold_acceptable: float = 3.0,
) -> Scorer:
    """Multi-model judge scorer for generation tasks.

    Args:
        judge_models: List of model IDs to use as judges
        threshold_excellent: Score >= this is CORRECT
        threshold_acceptable: Score >= this is PARTIAL

    Returns:
        Scorer that aggregates scores from multiple judges
    """

    async def score(state: TaskState, target: Target) -> Score:
        question = state.input_text
        answer = state.output.completion if state.output else ""
        target_text = target.text if target else ""

        if not answer:
            return Score(
                value=INCORRECT,
                answer=answer,
                explanation="No response generated",
            )

        result = await multi_model_judge(
            question=question,
            answer=answer,
            target=target_text,
            judge_models=judge_models,
        )

        mean_score = result["mean_score"]

        if mean_score is None:
            return Score(
                value=INCORRECT,
                answer=answer,
                explanation="All judges failed to provide scores",
                metadata={"explanations": result["explanations"]},
            )

        # Determine correctness category
        if mean_score >= threshold_excellent:
            value = CORRECT
        elif mean_score >= threshold_acceptable:
            value = PARTIAL
        else:
            value = INCORRECT

        # Build explanation
        score_summary = ", ".join(
            f"{model.split('/')[-1]}: {s}"
            for model, s in zip(judge_models, result["scores"])
        )

        explanation = (
            f"Mean: {mean_score:.2f} (std: {result['std_dev']:.2f})\n"
            f"Scores: {score_summary}\n"
            f"Agreement: {result['agreement']:.0%}"
        )

        return Score(
            value=value,
            answer=answer,
            explanation=explanation,
            metadata={
                "mean_score": mean_score,
                "individual_scores": result["scores"],
                "std_dev": result["std_dev"],
                "agreement": result["agreement"],
                "judge_models": judge_models,
                "explanations": result["explanations"],
            },
        )

    return score


# Convenience function for direct use
def get_multi_judge_scorer(
    models: list[str] = DEFAULT_JUDGE_MODELS,
) -> Scorer:
    """Get a multi-model judge scorer with specified models."""
    return multi_judge_scorer(judge_models=models)
