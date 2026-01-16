"""TreatyBench evaluation tasks."""

# Core tasks
from .classification import treaty_classification
from .extraction import obligation_extraction
from .generation import (
    treaty_generation,
    treaty_completion,
    treaty_generation_multijudge,
    treaty_completion_multijudge,
)

# New tasks
from .ambiguity import ambiguity_detection
from .cross_reference import cross_reference_resolution
from .revision import adversarial_revision

# Hard variants
from .hard_variants import (
    classification_hard,
    extraction_hard,
    generation_hard,
    generation_hard_multijudge,
)

__all__ = [
    # Core tasks (A1, A2, B1, B2)
    "treaty_classification",
    "obligation_extraction",
    "treaty_generation",
    "treaty_completion",
    # Multi-judge variants
    "treaty_generation_multijudge",
    "treaty_completion_multijudge",
    # New tasks (A3, A4, B3)
    "ambiguity_detection",
    "cross_reference_resolution",
    "adversarial_revision",
    # Hard variants
    "classification_hard",
    "extraction_hard",
    "generation_hard",
    "generation_hard_multijudge",
]
