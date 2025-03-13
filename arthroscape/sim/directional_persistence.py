# arthroscape/sim/directional_persistence.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import math
import logging

class DirectionalPersistenceStrategy(ABC):
    @abstractmethod
    def adjust_heading(self,
                       prev_heading: float,
                       computed_heading: float,
                       odor_left: float,
                       odor_right: float,
                       config: "SimulationConfig",
                       rng: np.random.Generator) -> float:
        """
        Given the previous heading, a computed heading (from turning logic),
        and current odor sensor readings, return an adjusted heading.
        """
        pass

class FixedBlendPersistence(DirectionalPersistenceStrategy):
    def __init__(self, alpha: float = 0.5):
        """
        :param alpha: Weight for the previous heading (0.0 means no persistence,
                      1.0 means full persistence).
        """
        self.alpha = alpha

    def adjust_heading(self, prev_heading: float, computed_heading: float,
                       odor_left: float, odor_right: float,
                       config: "SimulationConfig", rng: np.random.Generator) -> float:
        if self.alpha == 0.0:
            return computed_heading
        elif self.alpha == 1.0:
            return prev_heading
        # Simple fixed blending.
        new_heading = (1 - self.alpha) * computed_heading + self.alpha * prev_heading
        return new_heading


class OdorDifferenceWeightedPersistence(DirectionalPersistenceStrategy):
    def __init__(self, alpha_min: float = 0.3, alpha_max: float = 0.7):
        """
        :param alpha_min: Minimum weight for the previous heading.
        :param alpha_max: Maximum weight for the previous heading.
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def adjust_heading(self, prev_heading: float, computed_heading: float,
                       odor_left: float, odor_right: float,
                       config: "SimulationConfig", rng: np.random.Generator) -> float:
        # Example: When the difference in odor is small (i.e. ambiguous), use more persistence.
        odor_diff = abs(odor_left - odor_right)
        # Normalize odor_diff to a value between 0 and 1; adjust scaling as needed.
        persistence_factor = np.clip(1 - odor_diff / (odor_left + odor_right + 1e-6), 0, 1)
        # Optionally mix with a baseline alpha.
        alpha = self.alpha_min + persistence_factor * (self.alpha_max - self.alpha_min)
        new_heading = (1 - alpha) * computed_heading + alpha * prev_heading
        return new_heading


class AvgOdorWeightedPersistence(DirectionalPersistenceStrategy):
    def __init__(self, alpha_min: float = 0.3, alpha_max: float = 0.7):
        """
        :param alpha_min: Minimum weight for the previous heading.
        :param alpha_max: Maximum weight for the previous heading.
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def adjust_heading(self, prev_heading: float, computed_heading: float,
                       odor_left: float, odor_right: float,
                       config: "SimulationConfig", rng: np.random.Generator) -> float:
        # Example: When the average odor is high, use more persistence.
        avg_odor = 0.5 * (odor_left + odor_right)
        # Normalize avg_odor to a value between 0 and 1; adjust scaling as needed.
        persistence_factor = np.clip(avg_odor, 0, 1)
        # Optionally mix with a baseline alpha.
        alpha = self.alpha_min + persistence_factor * (self.alpha_max - self.alpha_min)
        new_heading = (1 - alpha) * computed_heading + alpha * prev_heading
        return new_heading