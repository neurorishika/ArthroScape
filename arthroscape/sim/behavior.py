# arthroscape/sim/behavior.py
"""
Behavior module for the ArthroScape simulation.

This module defines the `BehaviorAlgorithm` abstract base class and its concrete implementation
`DefaultBehavior`. These classes determine how agents update their state (stop/walk) and
heading based on sensory inputs (odor) and internal parameters.
"""

import numpy as np
from abc import ABC, abstractmethod
from .config import SimulationConfig
from .directional_persistence import DirectionalPersistenceStrategy


class BehaviorAlgorithm(ABC):
    """
    Abstract base class for behavioral algorithms.

    A behavior algorithm defines how an agent transitions between states (e.g., stop vs. walk)
    and how it updates its heading based on sensory input and environmental context.
    """

    @abstractmethod
    def update_state(
        self, prev_state: int, config: SimulationConfig, rng: np.random.Generator
    ) -> int:
        """
        Update the agent's state (e.g., 0 for stop, 1 for walk).

        Args:
            prev_state (int): The current state of the agent.
            config (SimulationConfig): The simulation configuration.
            rng (np.random.Generator): Random number generator.

        Returns:
            int: The new state of the agent.
        """
        pass

    @abstractmethod
    def update_heading(
        self,
        prev_heading: float,
        odor_left: float,
        odor_right: float,
        at_wall: bool,
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> float:
        """
        Update the agent's heading.

        Args:
            prev_heading (float): The current heading in radians.
            odor_left (float): Perceived odor concentration on the left antenna.
            odor_right (float): Perceived odor concentration on the right antenna.
            at_wall (bool): True if the agent is currently at a wall/obstacle.
            config (SimulationConfig): The simulation configuration.
            rng (np.random.Generator): Random number generator.

        Returns:
            float: The new heading in radians.
        """
        pass


class DefaultBehavior(BehaviorAlgorithm):
    """
    Default implementation of the behavioral algorithm.

    This behavior models:
    1.  Stochastic transitions between stop (0) and walk (1) states.
    2.  Heading updates driven by odor asymmetry (chemotaxis), random diffusion, and wall avoidance.
    3.  Directional persistence to smooth out turns.
    """

    def update_state(
        self, prev_state: int, config: SimulationConfig, rng: np.random.Generator
    ) -> int:
        """
        Update the agent's state based on transition probabilities defined in the config.

        Args:
            prev_state (int): The current state (0: stop, 1: walk).
            config (SimulationConfig): Simulation configuration containing transition rates.
            rng (np.random.Generator): Random number generator.

        Returns:
            int: The new state (0 or 1).
        """
        if prev_state == 0:
            return 1 if rng.random() < config.rate_stop_to_walk_per_frame else 0
        else:
            return 0 if rng.random() < config.rate_walk_to_stop_per_frame else 1

    def update_heading(
        self,
        prev_heading: float,
        odor_left: float,
        odor_right: float,
        at_wall: bool,
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> float:
        """
        Update the agent's heading based on odor gradients and random fluctuations.

        The logic includes:
        - Determining turn direction based on odor gradient (left vs. right).
        - Applying an error rate to occasionally turn against the gradient.
        - Turning if at a wall.
        - Turning probabilistically based on base turn rate and odor asymmetry.
        - Adding rotational diffusion (noise).
        - Applying directional persistence.

        Args:
            prev_heading (float): Current heading in radians.
            odor_left (float): Odor concentration at left sensor.
            odor_right (float): Odor concentration at right sensor.
            at_wall (bool): Whether the agent is colliding with a wall.
            config (SimulationConfig): Simulation configuration.
            rng (np.random.Generator): Random number generator.

        Returns:
            float: The updated heading in radians.
        """
        # if odor is close to zero, set it to zero
        if abs(odor_left) < 1e-6:
            odor_left = 0
        if abs(odor_right) < 1e-6:
            odor_right = 0
        turn_direction = (
            1
            if odor_left > odor_right
            else (-1 if odor_left < odor_right else rng.choice([-1, 1]))
        )
        # flip based on error rate
        if rng.random() < config.error_rate_per_frame:
            turn_direction *= -1
        if at_wall:
            turn_angle = config.turn_angle_sampler()
            new_heading = (
                prev_heading
                + turn_direction * turn_angle
                + rng.normal(0, config.rotation_diffusion)
            )
        elif rng.random() < (
            config.turn_rate_per_frame
            + abs(odor_left - odor_right) * config.asymmetry_factor_per_frame
        ):
            turn_angle = config.turn_angle_sampler() * (
                1 + config.odor_driven_turn_scaler * (odor_left + odor_right) / 2
            )
            new_heading = (
                prev_heading
                + turn_direction * turn_angle
                + rng.normal(0, config.rotation_diffusion)
            )
        else:
            new_heading = prev_heading + rng.normal(0, config.rotation_diffusion)

        # Now adjust the heading using the persistence strategy.
        adjusted_heading = config.directional_persistence_strategy.adjust_heading(
            prev_heading, new_heading, odor_left, odor_right, config, rng
        )
        return adjusted_heading
