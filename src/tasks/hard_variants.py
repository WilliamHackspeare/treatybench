"""
Hard Variants of Core Tasks

A1-Hard: Multi-domain classification (provisions blending domains)
A2-Hard: Complex extraction (nested conditionals, multiple actors)
B2-Hard: Novel scenario generation (emerging domains like AI weapons)
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate, system_message
from inspect_ai.scorer import match, model_graded_qa

# Import scorers - use sys.path hack for direct file execution
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.scorers import multi_judge_scorer
from src.scorers import obligation_f1_scorer

DATA_DIR = Path(__file__).parent.parent / "data"

# =============================================================================
# A1-Hard: Multi-Domain Classification
# =============================================================================

CLASSIFICATION_HARD_SYSTEM = """You are an expert in international law with deep knowledge of treaty domains.

Classify the following treaty provision into its PRIMARY domain. Note that some provisions
may touch on multiple domains - identify the PRIMARY focus based on the main subject matter
and obligations established.

Domains:
- arms_control: Weapons regulation, disarmament, non-proliferation, verification
- trade: Commerce, tariffs, services, intellectual property, market access
- environment: Climate, biodiversity, pollution, natural resources
- human_rights: Civil/political rights, economic/social rights, non-discrimination
- ai_governance: AI regulation, algorithmic systems, data protection, automated decisions

Some provisions blend domains (e.g., trade sanctions for human rights violations,
environmental provisions in trade agreements). Focus on the PRIMARY subject matter.

Respond with ONLY the domain name, nothing else."""


@task
def classification_hard():
    """Classify multi-domain treaty provisions (A1-Hard)."""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "classification_hard.jsonl")),
        solver=[
            system_message(CLASSIFICATION_HARD_SYSTEM),
            generate(),
        ],
        scorer=match(ignore_case=True),
    )


# =============================================================================
# A2-Hard: Complex Extraction
# =============================================================================

EXTRACTION_HARD_SYSTEM = """You are an expert in international law and treaty analysis.
Your task is to extract all obligations, rights, and prohibitions from complex treaty text.

These provisions contain:
- Multiple actors with different obligations
- Nested conditional statements
- Exceptions and carve-outs
- Temporal conditions

For each item, identify:
1. Type: "obligation", "right", or "prohibition"
2. Actor: Who bears this (be specific - "State Party", "Developed country Party", etc.)
3. Action: What they must/may/must not do
4. Condition: Any conditions or exceptions that apply (or "none")

Respond in JSON format:
[
  {"type": "obligation", "actor": "...", "action": "...", "condition": "..."},
  {"type": "prohibition", "actor": "...", "action": "...", "condition": "..."}
]

Be thorough - these provisions often contain 3-6 distinct obligations.
Respond with ONLY the JSON array, no additional text."""


@task
def extraction_hard():
    """Extract obligations from complex provisions (A2-Hard)."""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "extraction_hard.jsonl")),
        solver=[
            system_message(EXTRACTION_HARD_SYSTEM),
            generate(),
        ],
        scorer=obligation_f1_scorer(),
    )


# =============================================================================
# B2-Hard: Novel Scenario Generation
# =============================================================================

GENERATION_HARD_SYSTEM = """You are an expert treaty drafter with deep knowledge of international law.
Your task is to generate formal treaty provisions for novel or emerging scenarios.

These scenarios may involve:
- Emerging technologies (AI weapons, autonomous systems, cyber operations)
- Cross-domain issues (climate-security nexus, trade-human rights)
- Novel governance challenges (space resources, digital sovereignty)

Guidelines:
- Use formal treaty language ("shall", "undertakes to", "State Party")
- Be specific and implementable despite the novel subject matter
- Draw on existing treaty patterns where applicable
- Address verification and enforcement challenges
- Consider both rights and obligations of relevant actors

Generate a complete, legally sound provision."""

GENERATION_HARD_RUBRIC = """
You are evaluating the quality of a treaty provision for a novel/emerging scenario.

SPECIFICATION (what was requested):
{question}

GENERATED PROVISION:
{answer}

Rate the generated provision on a scale of 1-5:

5: Excellent - Innovative yet legally sound. Addresses the novel scenario with appropriate specificity, creates enforceable obligations, and draws appropriately on existing treaty patterns.

4: Good - Addresses the novel scenario adequately. Generally sound approach but may lack specificity in some areas or miss some enforcement considerations.

3: Adequate - Basic attempt at the novel scenario. Uses appropriate language but may be too generic or fail to address key challenges of the new domain.

2: Poor - Fails to adequately address the novel scenario. May be too vague, legally problematic, or ignore key aspects of the specification.

1: Failing - Does not address the novel scenario meaningfully. Incoherent, legally invalid, or misses the specification entirely.

Consider:
- Does it address the novel/emerging aspect appropriately?
- Is it legally sound and potentially enforceable?
- Does it provide sufficient specificity for implementation?
- Does it draw appropriately on existing treaty patterns?

First provide your numeric rating (1-5), then a brief justification.
Format: GRADE: [number]
"""


@task
def generation_hard():
    """Generate provisions for novel scenarios (B2-Hard)."""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "generation_hard.jsonl")),
        solver=[
            system_message(GENERATION_HARD_SYSTEM),
            generate(),
        ],
        scorer=model_graded_qa(
            template=GENERATION_HARD_RUBRIC,
            grade_pattern=r"GRADE:\s*([1-5])",
        ),
    )


@task
def generation_hard_multijudge():
    """Generate provisions for novel scenarios (B2-Hard) with multi-model judging."""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "generation_hard.jsonl")),
        solver=[
            system_message(GENERATION_HARD_SYSTEM),
            generate(),
        ],
        scorer=multi_judge_scorer(),
    )
