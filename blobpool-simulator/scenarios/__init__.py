"""
Simulation scenarios for sparse blobpool.

Each scenario demonstrates different aspects of the protocol or
specific experimental setups.
"""

from .basic_propagation import BasicPropagationScenario
from .adversarial import AdversarialScenario
from .stress_test import StressTestScenario

__all__ = [
    "BasicPropagationScenario",
    "AdversarialScenario",
    "StressTestScenario",
]
