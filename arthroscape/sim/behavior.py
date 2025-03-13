# arthroscape/sim/behavior.py
import numpy as np
from abc import ABC, abstractmethod
from .config import SimulationConfig
from .directional_persistence import DirectionalPersistenceStrategy
class BehaviorAlgorithm(ABC):
    @abstractmethod
    def update_state(self, prev_state: int, config: SimulationConfig, rng: np.random.Generator) -> int:
        pass

    @abstractmethod
    def update_heading(self, prev_heading: float, odor_left: float, odor_right: float,
                       at_wall: bool, config: SimulationConfig, rng: np.random.Generator) -> float:
        pass

class DefaultBehavior(BehaviorAlgorithm):
    def update_state(self, prev_state: int, config: SimulationConfig, rng: np.random.Generator) -> int:
        if prev_state == 0:
            return 1 if rng.random() < config.rate_stop_to_walk_per_frame else 0
        else:
            return 0 if rng.random() < config.rate_walk_to_stop_per_frame else 1

    def update_heading(self, prev_heading: float, odor_left: float, odor_right: float,
                       at_wall: bool, config: SimulationConfig, rng: np.random.Generator) -> float:
        turn_direction = 1 if odor_left > odor_right else (-1 if odor_left < odor_right else rng.choice([-1, 1]))
        if at_wall:
            turn_angle = config.turn_angle_sampler()
            new_heading = prev_heading + turn_direction * turn_angle + rng.normal(0, config.rotation_diffusion)
        elif rng.random() < (config.turn_rate_per_frame +
                             abs(odor_left - odor_right) * config.asymmetry_factor_per_frame):
            turn_angle = config.turn_angle_sampler()
            new_heading = prev_heading + turn_direction * turn_angle + rng.normal(0, config.rotation_diffusion)
        else:
            new_heading = prev_heading + rng.normal(0, config.rotation_diffusion)
        
        # Now adjust the heading using the persistence strategy.
        adjusted_heading = config.directional_persistence_strategy.adjust_heading(
            prev_heading, new_heading, odor_left, odor_right, config, rng
        )
        return adjusted_heading
