# arthroscape/sim/odor_release.py
"""
Odor release module for the ArthroScape simulation.

This module defines strategies for how agents release odor into the environment.
It includes the `OdorReleaseStrategy` abstract base class and implementations for
constant release and no release. It also handles the generation of Gaussian kernels
for odor deposition.
"""

from typing import List, Tuple, Sequence
from abc import ABC, abstractmethod
from .config import SimulationConfig
import numpy as np
from functools import lru_cache
from scipy.ndimage import shift  # Added for subpixel shifting


@lru_cache(maxsize=32)
def _get_normalized_gaussian_kernel(
    sigma: float, kernel_size: int, resolution: float
) -> np.ndarray:
    """
    Compute and return a normalized Gaussian kernel.

    The kernel is computed over a square grid of size kernel_size x kernel_size (assumed odd).
    Sigma is expressed in mm; resolution is mm per grid cell.

    Args:
        sigma (float): Standard deviation of the Gaussian in mm.
        kernel_size (int): Size of the kernel grid (number of cells).
        resolution (float): Grid resolution in mm/cell.

    Returns:
        np.ndarray: A 2D normalized Gaussian kernel.
    """
    half_size = kernel_size // 2
    x = np.linspace(-half_size * resolution, half_size * resolution, kernel_size)
    y = np.linspace(-half_size * resolution, half_size * resolution, kernel_size)
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


class DepositInstruction:
    """
    A data structure representing a request to deposit odor.

    It contains all necessary information to generate the deposition kernel and place it
    relative to the agent.
    """

    def __init__(
        self,
        offset: Tuple[float, float],
        intensity: float,
        sigma: float,
        kernel_size: int = None,
    ):
        """
        Initialize a deposit instruction.

        Args:
            offset (Tuple[float, float]): (dx, dy) offset in the animal's local coordinate frame (mm).
            intensity (float): Overall intensity multiplier (amount of odor).
            sigma (float): Standard deviation for the Gaussian deposit (in mm).
            kernel_size (int, optional): Kernel size (number of grid cells, assumed odd).
                                         If None, will use config value.
        """
        self.offset = offset
        self.intensity = intensity
        self.sigma = sigma
        self.kernel_size = kernel_size

    def generate_kernel(self, config: SimulationConfig) -> np.ndarray:
        """
        Generate the deposition kernel.

        Retrieves a cached normalized kernel and scales it by the intensity.

        Args:
            config (SimulationConfig): Simulation configuration.

        Returns:
            np.ndarray: The scaled 2D deposition kernel.
        """
        ksize = (
            self.kernel_size
            if self.kernel_size is not None
            else config.deposit_kernel_size
        )
        normalized_kernel = _get_normalized_gaussian_kernel(
            self.sigma, ksize, config.grid_resolution
        )
        return normalized_kernel * self.intensity


class OdorReleaseStrategy(ABC):
    """
    Abstract base class for odor release strategies.

    A release strategy determines when and how much odor an agent releases into the environment.
    """

    @abstractmethod
    def release_odor(
        self,
        state: int,
        position: Tuple[float, float],
        heading: float,
        config: SimulationConfig,
        rng,
    ) -> List[DepositInstruction]:
        """
        Determine odor release for the current time step.

        Args:
            state (int): Current state of the agent.
            position (Tuple[float, float]): Current position (x, y).
            heading (float): Current heading in radians.
            config (SimulationConfig): Simulation configuration.
            rng (np.random.Generator): Random number generator.

        Returns:
            List[DepositInstruction]: A list of instructions for depositing odor.
        """
        pass


class DefaultOdorRelease(OdorReleaseStrategy):
    """
    A default strategy that releases no odor.
    """

    def release_odor(
        self,
        state: int,
        position: Tuple[float, float],
        heading: float,
        config: SimulationConfig,
        rng,
    ) -> List[DepositInstruction]:
        """
        Return an empty list (no odor release).
        """
        return []


class ConstantOdorRelease(OdorReleaseStrategy):
    """
    Deposits a constant amount of pheromone with a Gaussian spread at each step.
    """

    def __init__(
        self,
        config: SimulationConfig,
        deposit_amount: float = 0.5,
        sigma: float = None,
        kernel_size: int = None,
        deposit_offsets: Sequence[Tuple[float, float]] = None,
    ):
        """
        Initialize the constant odor release strategy.

        Args:
            config (SimulationConfig): Simulation configuration.
            deposit_amount (float): Amount of odor to deposit per step.
            sigma (float, optional): Standard deviation of the deposit Gaussian. Defaults to config value.
            kernel_size (int, optional): Size of the deposit kernel. Defaults to config value.
            deposit_offsets (Sequence[Tuple[float, float]], optional): List of offsets for deposition. Defaults to config value.
        """
        self.deposit_amount = deposit_amount
        self.sigma = sigma if sigma is not None else config.deposit_sigma
        self.kernel_size = (
            kernel_size if kernel_size is not None else config.deposit_kernel_size
        )
        self.deposit_offsets = (
            deposit_offsets
            if deposit_offsets is not None
            else config.odor_deposit_offsets
        )

    def release_odor(
        self,
        state: int,
        position: Tuple[float, float],
        heading: float,
        config: SimulationConfig,
        rng,
    ) -> List[DepositInstruction]:
        """
        Generate a list of DepositInstruction objects based on the provided offsets.

        Args:
            state (int): Current state of the animal.
            position (Tuple[float, float]): Current position of the animal.
            heading (float): Current heading of the animal.
            config (SimulationConfig): Simulation configuration.
            rng (np.random.Generator): Random number generator.

        Returns:
            List[DepositInstruction]: List of DepositInstruction objects.
        """
        offsets = (
            self.deposit_offsets
            if self.deposit_offsets is not None
            else config.odor_deposit_offsets
        )
        instructions = []
        for offset in offsets:
            instructions.append(
                DepositInstruction(
                    offset=offset,
                    intensity=self.deposit_amount,
                    sigma=self.sigma,
                    kernel_size=self.kernel_size,
                )
            )
        return instructions
