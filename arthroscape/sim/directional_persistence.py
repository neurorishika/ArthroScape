# arthroscape/sim/directional_persistence.py
"""
Directional persistence module for the ArthroScape simulation.

This module defines strategies for directional persistence, which allows agents to maintain
their heading over time, smoothing out instantaneous turns. This is crucial for modeling
realistic movement where inertia or behavioral persistence plays a role.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import math
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SimulationConfig


class DirectionalPersistenceStrategy(ABC):
    """
    Abstract base class for directional persistence strategies.

    A persistence strategy modifies the computed heading (based on immediate sensory input)
    by blending it with the previous heading or applying other persistence logic.
    """

    @abstractmethod
    def adjust_heading(
        self,
        prev_heading: float,
        computed_heading: float,
        odor_left: float,
        odor_right: float,
        config: "SimulationConfig",
        rng: np.random.Generator,
    ) -> float:
        """
        Adjust the heading based on persistence logic.

        Args:
            prev_heading (float): The agent's heading in the previous frame.
            computed_heading (float): The proposed new heading based on current sensory input.
            odor_left (float): Odor concentration at the left sensor.
            odor_right (float): Odor concentration at the right sensor.
            config (SimulationConfig): The simulation configuration.
            rng (np.random.Generator): Random number generator.

        Returns:
            float: The adjusted heading.
        """
        pass


class FixedBlendPersistence(DirectionalPersistenceStrategy):
    """
    A simple persistence strategy that linearly blends the previous and computed headings.

    The new heading is a weighted average:
    new_heading = (1 - alpha) * computed_heading + alpha * prev_heading
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize the FixedBlendPersistence strategy.

        Args:
            alpha (float): The persistence weight (0.0 to 1.0).
                           0.0 means no persistence (fully reactive).
                           1.0 means full persistence (no turning).
        """
        self.alpha = alpha

    def adjust_heading(
        self,
        prev_heading: float,
        computed_heading: float,
        odor_left: float,
        odor_right: float,
        config: "SimulationConfig",
        rng: np.random.Generator,
    ) -> float:
        """
        Apply fixed blending to the heading.

        Args:
            prev_heading (float): Previous heading.
            computed_heading (float): Proposed heading.
            odor_left (float): Left odor.
            odor_right (float): Right odor.
            config (SimulationConfig): Config.
            rng (np.random.Generator): RNG.

        Returns:
            float: The blended heading.
        """
        if self.alpha == 0.0:
            return computed_heading
        elif self.alpha == 1.0:
            return prev_heading
        # Simple fixed blending.
        new_heading = (1 - self.alpha) * computed_heading + self.alpha * prev_heading
        return new_heading


class OdorDifferenceWeightedPersistence(DirectionalPersistenceStrategy):
    """
    Persistence strategy weighted by odor difference.

    This strategy increases persistence when the odor signal is ambiguous (small difference),
    encouraging the agent to "commit" to a direction when signals are weak.
    """

    def __init__(self, alpha_min: float = 0.3, alpha_max: float = 0.7):
        """
        Initialize the OdorDifferenceWeightedPersistence strategy.

        Args:
            alpha_min (float): Minimum persistence weight (used when odor difference is large).
            alpha_max (float): Maximum persistence weight (used when odor difference is small).
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def adjust_heading(
        self,
        prev_heading: float,
        computed_heading: float,
        odor_left: float,
        odor_right: float,
        config: "SimulationConfig",
        rng: np.random.Generator,
    ) -> float:
        """
        Adjust heading with persistence inversely proportional to odor difference.

        Args:
            prev_heading (float): Previous heading.
            computed_heading (float): Proposed heading.
            odor_left (float): Left odor.
            odor_right (float): Right odor.
            config (SimulationConfig): Config.
            rng (np.random.Generator): RNG.

        Returns:
            float: The adjusted heading.
        """
        # Example: When the difference in odor is small (i.e. ambiguous), use more persistence.
        odor_diff = abs(odor_left - odor_right)
        # Normalize odor_diff to a value between 0 and 1; adjust scaling as needed.
        persistence_factor = np.clip(
            1 - odor_diff / (odor_left + odor_right + 1e-6), 0, 1
        )
        # Optionally mix with a baseline alpha.
        alpha = self.alpha_min + persistence_factor * (self.alpha_max - self.alpha_min)
        new_heading = (1 - alpha) * computed_heading + alpha * prev_heading
        return new_heading


class AvgOdorWeightedPersistence(DirectionalPersistenceStrategy):
    """
    Persistence strategy weighted by average odor concentration.

    This strategy increases persistence when the overall odor concentration is high.
    """

    def __init__(self, alpha_min: float = 0.3, alpha_max: float = 0.7):
        """
        Initialize the AvgOdorWeightedPersistence strategy.

        Args:
            alpha_min (float): Minimum persistence weight (used when odor is low).
            alpha_max (float): Maximum persistence weight (used when odor is high).
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def adjust_heading(
        self,
        prev_heading: float,
        computed_heading: float,
        odor_left: float,
        odor_right: float,
        config: "SimulationConfig",
        rng: np.random.Generator,
    ) -> float:
        """
        Adjust heading with persistence proportional to average odor.

        Args:
            prev_heading (float): Previous heading.
            computed_heading (float): Proposed heading.
            odor_left (float): Left odor.
            odor_right (float): Right odor.
            config (SimulationConfig): Config.
            rng (np.random.Generator): RNG.

        Returns:
            float: The adjusted heading.
        """
        # Example: When the average odor is high, use more persistence.
        avg_odor = 0.5 * (odor_left + odor_right)
        # Normalize avg_odor to a value between 0 and 1; adjust scaling as needed.
        persistence_factor = np.clip(avg_odor, 0, 1)
        # Optionally mix with a baseline alpha.
        alpha = self.alpha_min + persistence_factor * (self.alpha_max - self.alpha_min)
        new_heading = (1 - alpha) * computed_heading + alpha * prev_heading
        return new_heading
