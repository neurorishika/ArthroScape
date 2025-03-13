# arthroscape/sim/odor_release.py
from typing import List, Tuple
from abc import ABC, abstractmethod
from .config import SimulationConfig

class OdorReleaseStrategy(ABC):
    @abstractmethod
    def release_odor(self, state: int, position: Tuple[float, float], heading: float,
                     config: SimulationConfig, rng) -> List[Tuple[float, float, float]]:
        """
        Return a list of odor deposits, each as (dx, dy, odor_value)
        relative to the animal's position.
        """
        pass

class DefaultOdorRelease(OdorReleaseStrategy):
    def release_odor(self, state: int, position: Tuple[float, float], heading: float,
                     config: SimulationConfig, rng) -> List[Tuple[float, float, float]]:
        return []

class ConstantOdorRelease(OdorReleaseStrategy):
    """
    Always deposit a constant amount of pheromone at the fly’s center.
    """
    def __init__(self, deposit_amount: float = 0.5):
        self.deposit_amount = deposit_amount

    def release_odor(self, state: int, position: Tuple[float, float], heading: float,
                     config: SimulationConfig, rng) -> List[Tuple[float, float, float]]:
        return [(0, 0, self.deposit_amount)]
