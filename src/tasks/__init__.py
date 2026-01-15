"""TreatyBench evaluation tasks."""

from .classification import treaty_classification
from .extraction import obligation_extraction
from .generation import treaty_generation, treaty_completion

__all__ = [
    "treaty_classification",
    "obligation_extraction",
    "treaty_generation",
    "treaty_completion",
]
