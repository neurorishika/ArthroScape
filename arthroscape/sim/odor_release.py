# arthroscape/sim/odor_release.py
from typing import List, Tuple, Sequence
from abc import ABC, abstractmethod
from .config import SimulationConfig
import numpy as np

class DepositInstruction:
    def __init__(self, offset: Tuple[float, float], intensity: float, sigma: float, kernel_size: int = None):
        """
        :param offset: (dx, dy) offset in the animal's local coordinate frame.
        :param intensity: Overall intensity multiplier.
        :param sigma: Standard deviation for the Gaussian deposit (in mm).
        :param kernel_size: Kernel size (number of grid cells, assumed odd). If None, will use config value.
        """
        self.offset = offset
        self.intensity = intensity
        self.sigma = sigma
        self.kernel_size = kernel_size

    def generate_kernel(self, config: SimulationConfig) -> np.ndarray:
        # Use provided kernel_size or default from config.
        ksize = self.kernel_size if self.kernel_size is not None else config.deposit_kernel_size
        half_size = ksize // 2
        resolution = config.grid_resolution
        x = np.linspace(-half_size * resolution, half_size * resolution, ksize)
        y = np.linspace(-half_size * resolution, half_size * resolution, ksize)
        X, Y = np.meshgrid(x, y)
        kernel = np.exp(-(X**2 + Y**2) / (2 * self.sigma**2))
        kernel = kernel / np.sum(kernel) * self.intensity
        return kernel

class OdorReleaseStrategy(ABC):
    @abstractmethod
    def release_odor(self, state: int, position: Tuple[float, float], heading: float,
                     config: SimulationConfig, rng) -> List[DepositInstruction]:
        pass

class DefaultOdorRelease(OdorReleaseStrategy):
    def release_odor(self, state: int, position: Tuple[float, float], heading: float,
                     config: SimulationConfig, rng) -> List[DepositInstruction]:
        return []

class ConstantOdorRelease(OdorReleaseStrategy):
    """
    Deposits a constant amount of pheromone with a Gaussian spread at each step.
    Uses deposit_offsets if provided; if not, uses config.odor_deposit_offsets.
    """
    def __init__(self, deposit_amount: float = 0.5, sigma: float = None,
                 kernel_size: int = None, deposit_offsets: Sequence[Tuple[float, float]] = None):
        self.deposit_amount = deposit_amount
        # If sigma not provided, use the config's default via later lookup.
        self.sigma = sigma if sigma is not None else 5.0
        self.kernel_size = kernel_size  # can be None to use config value
        self.deposit_offsets = deposit_offsets  # if None, will use config.odor_deposit_offsets in release_odor

    def release_odor(self, state: int, position: Tuple[float, float], heading: float,
                     config: SimulationConfig, rng) -> List[DepositInstruction]:
        offsets = self.deposit_offsets if self.deposit_offsets is not None else config.odor_deposit_offsets
        instructions = []
        for offset in offsets:
            instructions.append(DepositInstruction(offset=offset, intensity=self.deposit_amount,
                                                     sigma=self.sigma, kernel_size=self.kernel_size))
        return instructions
