"""
A1: Treaty Provision Classification Task

Classify treaty provisions by domain:
- arms_control
- trade
- environment
- human_rights
- ai_governance
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.scorer import match, accuracy, Metric
from inspect_ai.solver import generate, system_message

SYSTEM_PROMPT = """You are an expert in international law and treaty analysis.
Your task is to classify treaty provisions into their appropriate domain.

Available domains:
- arms_control: Nuclear weapons, chemical weapons, biological weapons, disarmament, verification
- trade: Tariffs, market access, goods, services, intellectual property
- environment: Climate, emissions, biodiversity, pollution, conservation
- human_rights: Civil liberties, dignity, discrimination, torture, fundamental freedoms
- ai_governance: Artificial intelligence, algorithms, automated systems, AI safety

Respond with ONLY the domain name, nothing else."""

DATA_DIR = Path(__file__).parent.parent / "data"


@task
def treaty_classification():
    """Classify treaty provisions by domain."""
    return Task(
        dataset=json_dataset(str(DATA_DIR / "classification.jsonl")),
        solver=[
            system_message(SYSTEM_PROMPT),
            generate(),
        ],
        scorer=match(ignore_case=True),
    )
