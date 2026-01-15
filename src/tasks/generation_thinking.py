"""
B1 & B2: Treaty Generation Tasks with Extended Thinking

Thinking-enabled variants for comparing reasoning effects.
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate, system_message
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import model_graded_qa

COMPLETION_SYSTEM = """You are an expert treaty drafter with deep knowledge of international law.
Your task is to complete partial treaty provisions by filling in missing text.

Guidelines:
- Use formal treaty language (e.g., "shall", "undertakes to", "State Party")
- Maintain consistency with the surrounding text
- Ensure the completion is legally precise and implementable
- Match the style and tone of the existing provision"""

GENERATION_SYSTEM = """You are an expert treaty drafter with deep knowledge of international law.
Your task is to generate formal treaty provisions based on specifications.

Guidelines:
- Use formal treaty language (e.g., "shall", "undertakes to", "State Party")
- Be specific and implementable
- Address all requirements in the specification
- Follow standard treaty drafting conventions
- Ensure legal precision and clarity"""

GENERATION_RUBRIC = """
You are evaluating the quality of a generated treaty provision.

SPECIFICATION (what was requested):
{question}

GENERATED PROVISION:
{answer}

Rate the generated provision on a scale of 1-5:

5: Excellent - Could serve as viable treaty language. Legally precise, addresses all requirements, uses appropriate formal language.

4: Good - Addresses requirements with minor gaps. Generally sound but could benefit from refinement.

3: Adequate - But generic or vague. Meets basic requirements but lacks specificity or precision.

2: Poor - Misses key requirements or contains significant issues.

1: Failing - Incoherent, legally invalid, or completely misses the specification.

Consider:
- Legal soundness: Is the language legally valid and enforceable?
- Specificity: Are terms and obligations clearly defined?
- Implementability: Could this be practically implemented?
- Completeness: Does it address all aspects of the specification?

First provide your numeric rating (1-5), then a brief justification.
Format: GRADE: [number]
"""

DATA_DIR = Path(__file__).parent.parent / "data"

# Extended thinking configuration
THINKING_CONFIG = GenerateConfig(
    reasoning_effort="high",  # For models that support it
)


@task
def treaty_completion_thinking():
    """Complete partial treaty provisions with extended thinking (B1)."""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "completion.jsonl")),
        solver=[
            system_message(COMPLETION_SYSTEM),
            generate(config=THINKING_CONFIG),
        ],
        scorer=model_graded_qa(
            template=GENERATION_RUBRIC,
            grade_pattern=r"GRADE:\s*([1-5])",
        ),
    )


@task
def treaty_generation_thinking():
    """Generate treaty provisions with extended thinking (B2)."""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "generation.jsonl")),
        solver=[
            system_message(GENERATION_SYSTEM),
            generate(config=THINKING_CONFIG),
        ],
        scorer=model_graded_qa(
            template=GENERATION_RUBRIC,
            grade_pattern=r"GRADE:\s*([1-5])",
        ),
    )
