"""
B3: Adversarial Provision Revision Task

Given a treaty provision with an identified flaw (loophole, ambiguity,
enforcement gap), revise the provision to fix the issue while maintaining
legal validity and consistency with treaty drafting conventions.
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate, system_message
from inspect_ai.scorer import model_graded_qa

SYSTEM_PROMPT = """You are an expert treaty drafter with deep knowledge of international law.
Your task is to revise flawed treaty provisions to address identified issues.

You will be given:
1. An original treaty provision
2. An identified flaw (loophole, ambiguity, enforcement gap, etc.)

Your task is to:
1. Revise the provision to fix the identified flaw
2. Maintain formal treaty language
3. Ensure the revision is legally sound and enforceable
4. Preserve the original intent while closing the gap

Guidelines:
- Use formal treaty language ("shall", "undertakes to", "State Party")
- Be specific and avoid introducing new ambiguities
- Ensure the revision is implementable and verifiable
- Make minimal changes necessary to fix the flaw

Output your revised provision directly. Do not include explanations or commentary."""

REVISION_RUBRIC = """
You are evaluating the quality of a revised treaty provision.

ORIGINAL PROVISION:
{question}

REVISED PROVISION:
{answer}

The revision was intended to fix an identified flaw in the original provision.

Rate the revised provision on a scale of 1-5:

5: Excellent - Effectively fixes the identified flaw while maintaining legal precision. The revision is enforceable, specific, and preserves the original intent.

4: Good - Addresses the flaw substantially but may have minor gaps. Generally sound but could benefit from refinement.

3: Adequate - Partially addresses the flaw but introduces new ambiguities or is overly broad/narrow.

2: Poor - Does not adequately fix the flaw, or the revision creates significant new problems.

1: Failing - Fails to address the flaw, is legally invalid, or fundamentally changes the provision's purpose.

Consider:
- Does it fix the identified flaw?
- Is it legally precise and enforceable?
- Does it maintain formal treaty language?
- Does it preserve the original intent?
- Does it avoid introducing new problems?

First provide your numeric rating (1-5), then a brief justification.
Format: GRADE: [number]
"""

DATA_DIR = Path(__file__).parent.parent / "data"


@task
def adversarial_revision():
    """Revise flawed treaty provisions to fix identified issues (B3)."""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "revision.jsonl")),
        solver=[
            system_message(SYSTEM_PROMPT),
            generate(),
        ],
        scorer=model_graded_qa(
            template=REVISION_RUBRIC,
            grade_pattern=r"GRADE:\s*([1-5])",
        ),
    )
