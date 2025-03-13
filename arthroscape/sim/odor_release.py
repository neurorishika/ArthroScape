# arthroscape/sim/odor_release.py
from typing import List, Tuple, Sequence
from abc import ABC, abstractmethod
from .config import SimulationConfig
import numpy as np
from functools import lru_cache
from scipy.ndimage import shift  # Added for subpixel shifting

@lru_cache(maxsize=32)
def _get_normalized_gaussian_kernel(sigma: float, kernel_size: int, resolution: float) -> np.ndarray:
    """
    Compute and return a normalized Gaussian kernel.
    The kernel is computed over a square grid of size kernel_size x kernel_size (assumed odd).
    Sigma is expressed in mm; resolution is mm per grid cell.
    """
    print(sigma, kernel_size, resolution)
    half_size = kernel_size // 2
    x = np.linspace(-half_size * resolution, half_size * resolution, kernel_size)
    y = np.linspace(-half_size * resolution, half_size * resolution, kernel_size)
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

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
        """
        Generate the deposition kernel by retrieving a cached normalized kernel and scaling it
        by the intensity.
        """
        ksize = self.kernel_size if self.kernel_size is not None else config.deposit_kernel_size
        normalized_kernel = _get_normalized_gaussian_kernel(self.sigma, ksize, config.grid_resolution)
        return normalized_kernel * self.intensity

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
    def __init__(self, config: SimulationConfig, deposit_amount: float = 0.5, sigma: float = None,
                 kernel_size: int = None, deposit_offsets: Sequence[Tuple[float, float]] = None):
        self.deposit_amount = deposit_amount
        self.sigma = sigma if sigma is not None else config.deposit_sigma
        self.kernel_size = kernel_size if kernel_size is not None else config.deposit_kernel_size
        self.deposit_offsets = deposit_offsets if deposit_offsets is not None else config.odor_deposit_offsets

    def release_odor(self, state: int, position: Tuple[float, float], heading: float,
                     config: SimulationConfig, rng) -> List[DepositInstruction]:
        """
        Generate a list of DepositInstruction objects based on the provided offsets.
        :param state: Current state of the animal.
        :param position: Current position of the animal.
        :param heading: Current heading of the animal.
        :param config: Simulation configuration.
        :param rng: Random number generator.
        :return: List of DepositInstruction objects.
        """
        offsets = self.deposit_offsets if self.deposit_offsets is not None else config.odor_deposit_offsets
        instructions = []
        for offset in offsets:
            instructions.append(DepositInstruction(offset=offset, intensity=self.deposit_amount,
                                                     sigma=self.sigma, kernel_size=self.kernel_size))
        return instructions
