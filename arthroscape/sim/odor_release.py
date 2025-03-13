# arthroscape/sim/odor_release.py
from typing import List, Tuple
from abc import ABC, abstractmethod
from .config import SimulationConfig
import numpy as np

class DepositInstruction:
    def __init__(self, offset: Tuple[float, float], intensity: float, sigma: float, kernel_size: int):
        """
        :param offset: (dx, dy) offset in the animal's local coordinate frame.
        :param intensity: Overall intensity multiplier.
        :param sigma: Standard deviation for the Gaussian deposit (in mm).
        :param kernel_size: Size of the kernel (number of grid cells, assumed odd).
        """
        self.offset = offset
        self.intensity = intensity
        self.sigma = sigma
        self.kernel_size = kernel_size

    def generate_kernel(self, config: SimulationConfig) -> np.ndarray:
        """
        Generate a Gaussian kernel scaled by the intensity.
        """
        resolution = config.grid_resolution  # mm per cell
        half_size = self.kernel_size // 2
        # Create coordinate grid (in mm) centered at zero.
        x = np.linspace(-half_size * resolution, half_size * resolution, self.kernel_size)
        y = np.linspace(-half_size * resolution, half_size * resolution, self.kernel_size)
        X, Y = np.meshgrid(x, y)
        kernel = np.exp(-(X**2 + Y**2) / (2 * self.sigma**2))
        kernel = kernel / np.sum(kernel) * self.intensity
        return kernel

class OdorReleaseStrategy(ABC):
    @abstractmethod
    def release_odor(self, state: int, position: Tuple[float, float], heading: float,
                     config: SimulationConfig, rng) -> List[DepositInstruction]:
        """
        Return a list of deposit instructions.
        """
        pass

class DefaultOdorRelease(OdorReleaseStrategy):
    def release_odor(self, state: int, position: Tuple[float, float], heading: float,
                     config: SimulationConfig, rng) -> List[DepositInstruction]:
        return []

class ConstantOdorRelease(OdorReleaseStrategy):
    """
    Always deposit a constant amount of pheromone with a Gaussian spread.
    """
    def __init__(self, deposit_amount: float = 0.5, sigma: float = 1.0, kernel_size: int = 7,
                 deposit_offsets: List[Tuple[float, float]] = None):
        self.deposit_amount = deposit_amount
        self.sigma = sigma
        self.kernel_size = kernel_size
        # If no offsets provided, default deposit at the centroid.
        if deposit_offsets is None:
            self.deposit_offsets = [(0, 0)]
        else:
            self.deposit_offsets = deposit_offsets

    def release_odor(self, state: int, position: Tuple[float, float], heading: float,
                     config: SimulationConfig, rng) -> List[DepositInstruction]:
        instructions = []
        for offset in self.deposit_offsets:
            instructions.append(DepositInstruction(offset=offset, intensity=self.deposit_amount,
                                                     sigma=self.sigma, kernel_size=self.kernel_size))
        return instructions
