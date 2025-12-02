# arthroscape/sim/arena.py
"""
Arena module for the ArthroScape simulation.

This module defines the `Arena` abstract base class and its concrete implementation `GridArena`.
The arena manages the spatial environment, including the odor field (concentration grid) and
obstacles (walls). It handles coordinate conversions between world space (continuous) and
grid space (discrete), as well as odor diffusion and decay dynamics.
"""

import numpy as np
import math
import logging
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from .config import SimulationConfig
from scipy.signal import fftconvolve, convolve2d
from scipy.ndimage import gaussian_filter, uniform_filter
from numba import njit

logger = logging.getLogger(__name__)


class Arena(ABC):
    """
    Abstract base class for the simulation arena.

    The Arena defines the interface for querying odor concentrations, checking for obstacles,
    and updating the odor field.
    """

    @abstractmethod
    def get_odor(self, x: float, y: float) -> float:
        """
        Return the odor concentration at a specific location.

        Args:
            x (float): The x-coordinate in world units (mm).
            y (float): The y-coordinate in world units (mm).

        Returns:
            float: The odor concentration at (x, y).
        """
        pass

    @abstractmethod
    def is_free(self, x: float, y: float) -> bool:
        """
        Check if a location is free of obstacles.

        Args:
            x (float): The x-coordinate in world units (mm).
            y (float): The y-coordinate in world units (mm).

        Returns:
            bool: True if the location is free, False if it is blocked (e.g., by a wall).
        """
        pass

    @abstractmethod
    def update_odor(self, x: float, y: float, odor: float) -> None:
        """
        Deposit odor at a specific location.

        This method adds odor to the environment, typically used when an agent releases odor.

        Args:
            x (float): The x-coordinate in world units (mm).
            y (float): The y-coordinate in world units (mm).
            odor (float): The amount of odor to deposit.
        """
        pass

    @abstractmethod
    def update_odor_field(
        self, dt: float = 1.0, method: str = "gaussian_filter"
    ) -> None:
        """
        Update the odor field dynamics (diffusion and decay).

        Args:
            dt (float): Time step for the update. Default is 1.0.
            method (str): The method to use for diffusion ('gaussian_filter', 'fftconvolve', etc.).
                          Default is 'gaussian_filter'.
        """
        pass


class GridArena(Arena):
    """
    A concrete implementation of Arena using a 2D grid.

    The GridArena discretizes the world into a grid of cells. It supports odor diffusion,
    decay, and wall obstacles defined by a mask.

    Attributes:
        x_min (float): Minimum x-coordinate of the arena.
        x_max (float): Maximum x-coordinate of the arena.
        y_min (float): Minimum y-coordinate of the arena.
        y_max (float): Maximum y-coordinate of the arena.
        resolution (float): Size of each grid cell in world units (mm).
        nx (int): Number of grid cells in the x-direction.
        ny (int): Number of grid cells in the y-direction.
        odor_grid (np.ndarray): 2D array storing odor concentrations.
        wall_mask (np.ndarray): 2D boolean array where True indicates a wall/obstacle.
        diffusion_coefficient (float): Diffusion coefficient for odor dynamics.
        odor_decay_rate (float): Rate at which odor decays per time step.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: float,
        wall_mask: Optional[np.ndarray] = None,
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize the GridArena.

        Args:
            x_min (float): Minimum x-coordinate.
            x_max (float): Maximum x-coordinate.
            y_min (float): Minimum y-coordinate.
            y_max (float): Maximum y-coordinate.
            resolution (float): Grid resolution (mm/cell).
            wall_mask (Optional[np.ndarray]): Boolean mask for walls. If None, no walls are present.
            config (Optional[SimulationConfig]): Simulation configuration object to pull parameters from.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution
        self.nx = int(np.ceil((x_max - x_min) / resolution)) + 1
        self.ny = int(np.ceil((y_max - y_min) / resolution)) + 1
        self.odor_grid = np.zeros((self.ny, self.nx))
        if wall_mask is None:
            self.wall_mask = np.zeros((self.ny, self.nx), dtype=bool)
        else:
            if wall_mask.shape != (self.ny, self.nx):
                raise ValueError("Wall mask dimensions do not match grid.")
            self.wall_mask = wall_mask
        if config is not None:
            self.diffusion_coefficient = config.diffusion_coefficient
            self.odor_decay_rate = config.odor_decay_rate
        else:
            self.diffusion_coefficient = 0.1
            self.odor_decay_rate = 0.001

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int, float, float]:
        """
        Convert world coordinates to grid indices and fractional offsets.

        Args:
            x (float): World x-coordinate.
            y (float): World y-coordinate.

        Returns:
            Tuple[int, int, float, float]: (i, j, fi, fj) where:
                i (int): Grid row index.
                j (int): Grid column index.
                fi (float): Fractional offset in x (0 to 1).
                fj (float): Fractional offset in y (0 to 1).
        """
        gx = (x - self.x_min) / self.resolution
        gy = (y - self.y_min) / self.resolution
        j = int(np.floor(gx))
        i = int(np.floor(gy))
        fi = gx - j
        fj = gy - i
        return i, j, fi, fj

    def world_to_grid_vectorized(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert arrays of world coordinates to grid indices and fractional parts.

        Args:
            x (np.ndarray): NumPy array of x coordinates.
            y (np.ndarray): NumPy array of y coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple (i, j, fi, fj) where:
                i (np.ndarray): Integer grid row indices.
                j (np.ndarray): Integer grid column indices.
                fi (np.ndarray): Fractional parts in x.
                fj (np.ndarray): Fractional parts in y.
        """
        return world_to_grid_vectorized_numba(
            x, y, self.x_min, self.y_min, self.resolution, self.nx, self.ny
        )

    def get_odor(self, x: float, y: float) -> float:
        """
        Get the odor concentration at a specific world location using bilinear interpolation.

        Args:
            x (float): World x-coordinate.
            y (float): World y-coordinate.

        Returns:
            float: Interpolated odor concentration. Returns 0.0 if outside the arena.
        """
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return 0.0
        i, j, fi, fj = self._world_to_grid(x, y)
        i1 = min(i, self.ny - 2)
        j1 = min(j, self.nx - 2)
        Q11 = self.odor_grid[i1, j1]
        Q21 = self.odor_grid[i1, j1 + 1]
        Q12 = self.odor_grid[i1 + 1, j1]
        Q22 = self.odor_grid[i1 + 1, j1 + 1]
        return (
            Q11 * (1 - fi) * (1 - fj)
            + Q21 * fi * (1 - fj)
            + Q12 * (1 - fi) * fj
            + Q22 * fi * fj
        )

    def get_odor_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Vectorized version of get_odor using bilinear interpolation.

        Args:
            xs (np.ndarray): NumPy array of x coordinates.
            ys (np.ndarray): NumPy array of y coordinates.

        Returns:
            np.ndarray: Array of odor concentrations corresponding to the input coordinates.
        """
        return get_odor_vectorized_numba(
            xs,
            ys,
            self.odor_grid,
            self.x_min,
            self.y_min,
            self.resolution,
            self.nx,
            self.ny,
        )

    def is_free(self, x: float, y: float) -> bool:
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return False
        i, j, _, _ = self._world_to_grid(x, y)
        i = np.clip(i, 0, self.ny - 1)
        j = np.clip(j, 0, self.nx - 1)
        return not self.wall_mask[i, j]

    def is_free_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Vectorized check if locations are free of obstacles.

        Args:
            xs (np.ndarray): NumPy array of x coordinates.
            ys (np.ndarray): NumPy array of y coordinates.

        Returns:
            np.ndarray: Boolean NumPy array indicating free (True) or blocked (False) for each coordinate.
        """
        return is_free_vectorized_numba(
            xs,
            ys,
            self.wall_mask,
            self.x_min,
            self.y_min,
            self.resolution,
            self.nx,
            self.ny,
        )

    def update_odor(self, x: float, y: float, odor: float) -> None:
        """
        Deposit odor at a specific location.

        Args:
            x (float): The x-coordinate in world units (mm).
            y (float): The y-coordinate in world units (mm).
            odor (float): The amount of odor to deposit.
        """
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return
        i, j, _, _ = self._world_to_grid(x, y)
        i = np.clip(i, 0, self.ny - 1)
        j = np.clip(j, 0, self.nx - 1)
        self.odor_grid[i, j] += odor

    def update_odor_vectorized(
        self, xs: np.ndarray, ys: np.ndarray, odors: np.ndarray
    ) -> None:
        """
        Vectorized update: deposit odor values at multiple positions.

        Args:
            xs (np.ndarray): 1D NumPy array of x coordinates.
            ys (np.ndarray): 1D NumPy array of y coordinates.
            odors (np.ndarray): 1D NumPy array of odor values.
        """
        self.odor_grid = update_odor_vectorized_numba(
            self.odor_grid,
            xs,
            ys,
            odors,
            self.x_min,
            self.y_min,
            self.resolution,
            self.nx,
            self.ny,
        )

    def deposit_odor_kernel(self, x: float, y: float, kernel: np.ndarray) -> None:
        """
        Deposit a single kernel (2D array) onto the odor grid at a given position.

        Args:
            x (float): The x-coordinate of the deposit center.
            y (float): The y-coordinate of the deposit center.
            kernel (np.ndarray): 2D array representing the odor distribution to add.
                                 Assumed to be square with odd dimensions.
        """
        ksize = kernel.shape[0]  # assume square kernel with odd dimensions
        half_size = ksize // 2
        i_center, j_center, _, _ = self._world_to_grid(x, y)
        i0 = max(i_center - half_size, 0)
        i1 = min(i_center + half_size + 1, self.ny)
        j0 = max(j_center - half_size, 0)
        j1 = min(j_center + half_size + 1, self.nx)
        ki0 = half_size - (i_center - i0)
        kj0 = half_size - (j_center - j0)
        ki1 = ki0 + (i1 - i0)
        kj1 = kj0 + (j1 - j0)
        self.odor_grid[i0:i1, j0:j1] += kernel[ki0:ki1, kj0:kj1]

    def deposit_odor_kernels_vectorized(
        self, xs: np.ndarray, ys: np.ndarray, kernel: np.ndarray
    ) -> None:
        """
        Vectorized deposition: deposit the same kernel at multiple positions.

        This method uses a vectorized world-to-grid conversion and then loops over the deposit positions.
        Note: Further vectorization would require advanced indexing and careful boundary handling.

        Args:
            xs (np.ndarray): 1D NumPy array of x coordinates for deposit centers.
            ys (np.ndarray): 1D NumPy array of y coordinates for deposit centers.
            kernel (np.ndarray): 2D NumPy array representing the kernel (assumed square, odd dimensions).
        """
        # get world to grid indices
        i, j, _, _ = self.world_to_grid_vectorized(xs, ys)
        self.odor_grid = deposit_odor_kernels_vectorized_numba(
            self.odor_grid, i, j, kernel, self.ny, self.nx
        )

    def _compute_diffusion_kernel(self, dt: float) -> Tuple[float, np.ndarray]:
        """
        Compute the Gaussian diffusion kernel for a given time step.

        Args:
            dt (float): Time step in seconds.

        Returns:
            Tuple[float, np.ndarray]: A tuple containing:
                - sigma_grid (float): The standard deviation in grid units.
                - kernel (np.ndarray): The computed 2D Gaussian kernel.
        """
        var = self.diffusion_coefficient * dt
        diffusion_sigma_mm = np.sqrt(var)
        sigma_grid = diffusion_sigma_mm / self.resolution
        kernel_size = int(2 * np.ceil(3 * sigma_grid)) + 1
        ax = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        ay = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        X, Y = np.meshgrid(ax, ay)
        kernel = np.exp(-(X**2 + Y**2) / (2 * sigma_grid**2))
        kernel /= np.sum(kernel)
        return sigma_grid, kernel

    def update_odor_field(self, dt: float = 1.0, method: str = "box_blur") -> None:
        """
        Update the odor field dynamics (diffusion and decay).

        Args:
            dt (float): Time step for the update. Default is 1.0.
            method (str): The method to use for diffusion. Options are:
                          - 'fft': FFT-based convolution (fast for large kernels).
                          - 'convolve2d': Scipy's convolve2d.
                          - 'gaussian_filter': Scipy's gaussian_filter.
                          - 'box_blur': Approximate Gaussian blur using multiple box filters (fastest).
                          Default is 'box_blur'.

        Raises:
            ValueError: If an unknown diffusion method is specified.
        """
        if self.diffusion_coefficient == 0:
            self.odor_grid *= 1 - self.odor_decay_rate
            if hasattr(self, "base_odor"):
                self.odor_grid = np.maximum(self.odor_grid, self.base_odor)
            return

        if not hasattr(self, "_diffusion_dt"):
            self._diffusion_dt = dt

        if not hasattr(self, "_diffusion_kernel") or self._diffusion_dt != dt:
            sigma_grid, kernel = self._compute_diffusion_kernel(dt)
            self._diffusion_sigma = sigma_grid
            self._diffusion_kernel = kernel
            self._diffusion_dt = dt
            logger.info(f"Computed and cached diffusion kernel for dt={dt}")

        if method == "fft":
            # FFT-based convolution
            if not hasattr(self, "_diffusion_kernel_fft"):
                self._diffusion_kernel_fft = np.fft.rfftn(
                    self._diffusion_kernel, s=self.odor_grid.shape
                )
                logger.info("Computed and cached FFT of diffusion kernel.")
            odor_fft = np.fft.rfftn(self.odor_grid)
            convolved = np.fft.irfftn(
                odor_fft * self._diffusion_kernel_fft, s=self.odor_grid.shape
            )
            self.odor_grid = convolved
        elif method == "convolve2d":
            self.odor_grid = convolve2d(
                self.odor_grid,
                self._diffusion_kernel,
                mode="same",
                boundary="fill",
                fillvalue=0,
            )
        elif method == "gaussian_filter":
            self.odor_grid = gaussian_filter(
                self.odor_grid, sigma=self._diffusion_sigma, mode="constant", cval=0
            )
        elif method == "box_blur":
            # Use uniform_filter to approximate Gaussian blur by applying it three times.
            # For a 1D box filter of width w, the variance is (w^2 - 1) / 12.
            # Here we set the box filter width such that its variance approximates that of our Gaussian.
            # A common heuristic is r = ceil(sigma * sqrt(3)), so kernel size = 2*r+1.
            r = int(np.ceil(self._diffusion_sigma * np.sqrt(3)))
            size = 2 * r + 1
            # Apply the box filter three times for a better approximation.
            self.odor_grid = uniform_filter(
                self.odor_grid, size=size, mode="constant", cval=0
            )
            self.odor_grid = uniform_filter(
                self.odor_grid, size=size, mode="constant", cval=0
            )
            self.odor_grid = uniform_filter(
                self.odor_grid, size=size, mode="constant", cval=0
            )
        else:
            raise ValueError(
                "Unknown diffusion method: choose 'fft', 'convolve2d', 'gaussian_filter', or 'box_blur'."
            )

        # Finally, apply odor decay.
        self.odor_grid *= 1 - self.odor_decay_rate

        # Optionally clamp the odor grid if needed.
        if hasattr(self, "base_odor"):
            self.odor_grid = np.maximum(self.odor_grid, self.base_odor)


#####################################################################
# Numba-accelerated helper functions for vectorized operations
#####################################################################


@njit
def world_to_grid_vectorized_numba(
    xs: np.ndarray,
    ys: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated conversion of world coordinates to grid indices and fractional parts.

    Args:
        xs (np.ndarray): 1D NumPy array of x coordinates.
        ys (np.ndarray): 1D NumPy array of y coordinates.
        x_min (float): Minimum x of the grid.
        y_min (float): Minimum y of the grid.
        resolution (float): Grid resolution.
        nx (int): Number of columns in grid.
        ny (int): Number of rows in grid.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of 1D NumPy arrays (i, j, fi, fj).
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = np.floor(gx).astype(np.int32)
    i = np.floor(gy).astype(np.int32)
    fi = gx - j
    fj = gy - np.floor(gy)
    return i, j, fi, fj


@njit
def get_odor_vectorized_numba(
    xs: np.ndarray,
    ys: np.ndarray,
    odor_grid: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Numba-accelerated bilinear interpolation for odor values.

    Args:
        xs (np.ndarray): 1D NumPy array of x coordinates.
        ys (np.ndarray): 1D NumPy array of y coordinates.
        odor_grid (np.ndarray): 2D NumPy array representing the odor grid.
        x_min (float): Minimum x of the grid.
        y_min (float): Minimum y of the grid.
        resolution (float): Grid resolution.
        nx (int): Number of columns in grid.
        ny (int): Number of rows in grid.

    Returns:
        np.ndarray: 1D NumPy array of interpolated odor values.
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = np.floor(gx).astype(np.int32)
    i = np.floor(gy).astype(np.int32)
    fi = gx - j
    fj = gy - np.floor(gy)
    # Clip indices for valid interpolation (assume grid has at least 2 cells in each dimension)
    i = np.clip(i, 0, ny - 2)
    j = np.clip(j, 0, nx - 2)
    result = np.empty(xs.shape, dtype=odor_grid.dtype)
    for idx in range(xs.size):
        Q11 = odor_grid[i[idx], j[idx]]
        Q21 = odor_grid[i[idx], j[idx] + 1]
        Q12 = odor_grid[i[idx] + 1, j[idx]]
        Q22 = odor_grid[i[idx] + 1, j[idx] + 1]
        result[idx] = (
            Q11 * (1 - fi[idx]) * (1 - fj[idx])
            + Q21 * fi[idx] * (1 - fj[idx])
            + Q12 * (1 - fi[idx]) * fj[idx]
            + Q22 * fi[idx] * fj[idx]
        )
    return result


@njit
def is_free_vectorized_numba(
    xs: np.ndarray,
    ys: np.ndarray,
    wall_mask: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Numba-accelerated check of free space for multiple positions.

    Args:
        xs (np.ndarray): 1D array of x coordinates.
        ys (np.ndarray): 1D array of y coordinates.
        wall_mask (np.ndarray): 2D boolean array indicating blocked cells.
        x_min (float): Minimum x of the grid.
        y_min (float): Minimum y of the grid.
        resolution (float): Grid resolution.
        nx (int): Number of columns.
        ny (int): Number of rows.

    Returns:
        np.ndarray: Boolean array (True if free).
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = np.floor(gx).astype(np.int32)
    i = np.floor(gy).astype(np.int32)
    i = np.clip(i, 0, ny - 1)
    j = np.clip(j, 0, nx - 1)
    result = np.empty(xs.shape, dtype=np.bool_)
    for idx in range(xs.size):
        result[idx] = not wall_mask[i[idx], j[idx]]
    return result


@njit
def update_odor_vectorized_numba(
    odor_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    odors: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Numba-accelerated update of the odor grid given multiple deposit positions.

    This function uses np.add.at to handle repeated indices.

    Args:
        odor_grid (np.ndarray): 2D array representing the odor grid.
        xs (np.ndarray): 1D array of x deposit positions.
        ys (np.ndarray): 1D array of y deposit positions.
        odors (np.ndarray): 1D array of odor values to add.
        x_min (float): Minimum x of the grid.
        y_min (float): Minimum y of the grid.
        resolution (float): Grid resolution.
        nx (int): Number of columns.
        ny (int): Number of rows.

    Returns:
        np.ndarray: The updated odor grid.
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = np.floor(gx).astype(np.int32)
    i = np.floor(gy).astype(np.int32)
    i = np.clip(i, 0, ny - 1)
    j = np.clip(j, 0, nx - 1)

    # Flatten indices for add.at
    flat_indices = i * nx + j
    odor_flat = odor_grid.ravel()
    # Numba doesn't support np.add.at directly in nopython mode efficiently for this case usually,
    # but let's check if we can use a loop.
    # Actually, for a simple loop:
    for idx in range(xs.size):
        odor_grid[i[idx], j[idx]] += odors[idx]

    return odor_grid


@njit
def _get_odor_vectorized_numba_dup(xs, ys, odor_grid, x_min, y_min, resolution, nx, ny):
    """
    Duplicate definition - this function is unused.

    Args:
        xs (np.ndarray): 1D NumPy array of x coordinates.
        ys (np.ndarray): 1D NumPy array of y coordinates.
        odor_grid (np.ndarray): 2D NumPy array representing the odor grid.
        x_min (float): Minimum x of the grid.
        y_min (float): Minimum y of the grid.
        resolution (float): Grid resolution.
        nx (int): Number of columns in grid.
        ny (int): Number of rows in grid.

    Returns:
        np.ndarray: 1D NumPy array of interpolated odor values.
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = np.floor(gx).astype(np.int32)
    i = np.floor(gy).astype(np.int32)
    fi = gx - j
    fj = gy - np.floor(gy)
    # Clip indices for valid interpolation (assume grid has at least 2 cells in each dimension)
    i = np.clip(i, 0, ny - 2)
    j = np.clip(j, 0, nx - 2)
    result = np.empty(xs.shape, dtype=odor_grid.dtype)
    for idx in range(xs.size):
        Q11 = odor_grid[i[idx], j[idx]]
        Q21 = odor_grid[i[idx], j[idx] + 1]
        Q12 = odor_grid[i[idx] + 1, j[idx]]
        Q22 = odor_grid[i[idx] + 1, j[idx] + 1]
        result[idx] = (
            Q11 * (1 - fi[idx]) * (1 - fj[idx])
            + Q21 * fi[idx] * (1 - fj[idx])
            + Q12 * (1 - fi[idx]) * fj[idx]
            + Q22 * fi[idx] * fj[idx]
        )
    return result


@njit
def _is_free_vectorized_numba_dup(xs, ys, wall_mask, x_min, y_min, resolution, nx, ny):
    """
    Duplicate definition - this function is unused.

    Args:
        xs (np.ndarray): 1D array of x coordinates.
        ys (np.ndarray): 1D array of y coordinates.
        wall_mask (np.ndarray): 2D boolean array indicating blocked cells.
        x_min (float): Minimum x of the grid.
        y_min (float): Minimum y of the grid.
        resolution (float): Grid resolution.
        nx (int): Number of columns.
        ny (int): Number of rows.

    Returns:
        np.ndarray: Boolean array (True if free).
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = np.floor(gx).astype(np.int32)
    i = np.floor(gy).astype(np.int32)
    i = np.clip(i, 0, ny - 1)
    j = np.clip(j, 0, nx - 1)
    result = np.empty(xs.shape, dtype=np.bool_)
    for idx in range(xs.size):
        result[idx] = not wall_mask[i[idx], j[idx]]
    return result


@njit
def _update_odor_vectorized_numba_dup(
    odor_grid, xs, ys, odors, x_min, y_min, resolution, nx, ny
):
    """
    Duplicate definition - this function is unused.

    Args:
        odor_grid (np.ndarray): 2D array representing the odor grid.
        xs (np.ndarray): 1D array of x deposit positions.
        ys (np.ndarray): 1D array of y deposit positions.
        odors (np.ndarray): 1D array of odor values to add.
        x_min (float): Minimum x of the grid.
        y_min (float): Minimum y of the grid.
        resolution (float): Grid resolution.
        nx (int): Number of columns.
        ny (int): Number of rows.

    Returns:
        np.ndarray: The updated odor grid.
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = np.floor(gx).astype(np.int32)
    i = np.floor(gy).astype(np.int32)
    for idx in range(xs.size):
        # Clip indices in case of deposits outside grid
        ii = i[idx]
        jj = j[idx]
        if ii < 0:
            ii = 0
        elif ii >= ny:
            ii = ny - 1
        if jj < 0:
            jj = 0
        elif jj >= nx:
            jj = nx - 1
        odor_grid[ii, jj] += odors[idx]
    return odor_grid


@njit
def deposit_odor_kernel_numba(
    odor_grid: np.ndarray,
    i_center: int,
    j_center: int,
    kernel: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    """
    Numba-accelerated deposition of a single kernel at a grid index with periodic wrapping.

    Args:
        odor_grid (np.ndarray): 2D array of the odor grid.
        i_center (int): Center row index for deposit.
        j_center (int): Center column index for deposit.
        kernel (np.ndarray): 2D array representing the deposition kernel.
        ny (int): Number of rows in odor_grid.
        nx (int): Number of columns in odor_grid.

    Returns:
        np.ndarray: Updated odor_grid.
    """
    ksize = kernel.shape[0]
    half_size = ksize // 2

    # Compute deposition boundaries on the grid.
    i0 = max(i_center - half_size, 0)
    i1 = min(i_center + half_size + 1, ny)
    j0 = max(j_center - half_size, 0)
    j1 = min(j_center + half_size + 1, nx)

    # Compute corresponding kernel indices.
    ki0 = half_size - (i_center - i0)
    kj0 = half_size - (j_center - j0)
    ki1 = ki0 + (i1 - i0)
    kj1 = kj0 + (j1 - j0)

    # Deposit the kernel on the grid.
    odor_grid[i0:i1, j0:j1] += kernel[ki0:ki1, kj0:kj1]
    return odor_grid


@njit
def deposit_odor_kernels_vectorized_numba(
    odor_grid: np.ndarray,
    i_centers: np.ndarray,
    j_centers: np.ndarray,
    kernel: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    """
    Numba-accelerated vectorized deposition of the same kernel at multiple grid indices.

    Args:
        odor_grid (np.ndarray): 2D array representing the odor grid.
        i_centers (np.ndarray): 1D array of center row indices (ints).
        j_centers (np.ndarray): 1D array of center column indices (ints).
        kernel (np.ndarray): 2D kernel array (assumed square, odd dimensions).
        ny (int): Number of rows in grid.
        nx (int): Number of columns in grid.

    Returns:
        np.ndarray: Updated odor_grid.
    """
    for idx in range(i_centers.size):
        odor_grid = deposit_odor_kernel_numba(
            odor_grid, i_centers[idx], j_centers[idx], kernel, ny, nx
        )
    return odor_grid


#####################################################################
# End of Numba helper functions
#####################################################################


class PeriodicSquareArena(GridArena):
    """
    A square arena with periodic (toroidal) boundary conditions.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: float,
        config: Optional[SimulationConfig] = None,
    ):
        super().__init__(
            x_min,
            x_max,
            y_min,
            y_max,
            resolution,
            wall_mask=np.zeros(
                (
                    int(np.ceil((y_max - y_min) / resolution)) + 1,
                    int(np.ceil((x_max - x_min) / resolution)) + 1,
                ),
                dtype=bool,
            ),
            config=config,
        )

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int, float, float]:
        gx = (x - self.x_min) / self.resolution
        gy = (y - self.y_min) / self.resolution
        j = int(np.floor(gx)) % self.nx
        i = int(np.floor(gy)) % self.ny
        fi = gx - np.floor(gx)
        fj = gy - np.floor(gy)
        return i, j, fi, fj

    def world_to_grid_vectorized(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert arrays of world coordinates to grid indices and fractional parts.

        Args:
            x (np.ndarray): NumPy array of x coordinates.
            y (np.ndarray): NumPy array of y coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of arrays
                (i, j, fi, fj) where i and j are integer grid indices, and fi and fj
                are the fractional parts.
        """
        return world_to_grid_periodic_vectorized_numba(
            x, y, self.x_min, self.y_min, self.resolution, self.nx, self.ny
        )

    def get_odor(self, x: float, y: float) -> float:
        i, j, fi, fj = self._world_to_grid(x, y)
        i1 = (i + 1) % self.ny
        j1 = (j + 1) % self.nx
        Q11 = self.odor_grid[i, j]
        Q21 = self.odor_grid[i, j1]
        Q12 = self.odor_grid[i1, j]
        Q22 = self.odor_grid[i1, j1]
        return (
            Q11 * (1 - fi) * (1 - fj)
            + Q21 * fi * (1 - fj)
            + Q12 * (1 - fi) * fj
            + Q22 * fi * fj
        )

    def get_odor_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Vectorized version of get_odor using bilinear interpolation.

        Args:
            xs (np.ndarray): NumPy array of x coordinates.
            ys (np.ndarray): NumPy array of y coordinates.

        Returns:
            np.ndarray: NumPy array of interpolated odor values.
        """
        return get_odor_periodic_vectorized_numba(
            xs,
            ys,
            self.odor_grid,
            self.x_min,
            self.y_min,
            self.resolution,
            self.nx,
            self.ny,
        )

    def is_free(self, x: float, y: float) -> bool:
        i, j, _, _ = self._world_to_grid(x, y)
        return not self.wall_mask[i, j]

    def is_free_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Vectorized version of is_free.

        Args:
            xs (np.ndarray): NumPy array of x coordinates.
            ys (np.ndarray): NumPy array of y coordinates.

        Returns:
            np.ndarray: Boolean NumPy array indicating free (True) or blocked (False)
                for each coordinate.
        """
        return is_free_periodic_vectorized_numba(
            xs,
            ys,
            self.wall_mask,
            self.x_min,
            self.y_min,
            self.resolution,
            self.nx,
            self.ny,
        )

    def update_odor(self, x: float, y: float, odor: float) -> None:
        i, j, _, _ = self._world_to_grid(x, y)
        self.odor_grid[i, j] += odor

    def update_odor_vectorized(
        self, xs: np.ndarray, ys: np.ndarray, odors: np.ndarray
    ) -> None:
        """
        Vectorized update: deposit odor values at multiple positions.

        Args:
            xs (np.ndarray): 1D NumPy array of x coordinates.
            ys (np.ndarray): 1D NumPy array of y coordinates.
            odors (np.ndarray): 1D NumPy array of odor values.
        """
        self.odor_grid = update_odor_periodic_vectorized_numba(
            self.odor_grid,
            xs,
            ys,
            odors,
            self.x_min,
            self.y_min,
            self.resolution,
            self.nx,
            self.ny,
        )

    def deposit_odor_kernel(self, x: float, y: float, kernel: np.ndarray) -> None:
        ksize = kernel.shape[0]
        half_size = ksize // 2
        i_center, j_center, _, _ = self._world_to_grid(x, y)
        ny, nx = self.odor_grid.shape
        for di in range(ksize):
            for dj in range(ksize):
                # Compute target indices with periodic wrapping
                i = (i_center - half_size + di) % ny
                j = (j_center - half_size + dj) % nx
                self.odor_grid[i, j] += kernel[di, dj]

    def deposit_odor_kernels_vectorized(
        self, xs: np.ndarray, ys: np.ndarray, kernel: np.ndarray
    ) -> None:
        """
        Vectorized deposition: deposit the same kernel at multiple positions.

        This method uses a vectorized world-to-grid conversion and then loops over
        the deposit positions. Note: Further vectorization would require advanced
        indexing and careful boundary handling.

        Args:
            xs (np.ndarray): 1D NumPy array of x coordinates for deposit centers.
            ys (np.ndarray): 1D NumPy array of y coordinates for deposit centers.
            kernel (np.ndarray): 2D NumPy array representing the kernel (assumed
                square, odd dimensions).
        """
        i, j, _, _ = self.world_to_grid_vectorized(xs, ys)
        self.odor_grid = deposit_odor_kernels_periodic_vectorized_numba(
            self.odor_grid, i, j, kernel, self.ny, self.nx
        )

    def update_odor_field(self, dt: float = 1.0, method: str = "box_blur") -> None:
        if self.diffusion_coefficient == 0:
            self.odor_grid *= 1 - self.odor_decay_rate
            if hasattr(self, "base_odor"):
                self.odor_grid = np.maximum(self.odor_grid, self.base_odor)
            return

        if not hasattr(self, "_diffusion_dt"):
            self._diffusion_dt = dt

        if not hasattr(self, "_diffusion_kernel") or self._diffusion_dt != dt:
            sigma, kernel = self._compute_diffusion_kernel(dt)
            self._diffusion_sigma = sigma
            self._diffusion_kernel = kernel
            self._diffusion_dt = dt
            logger.info(f"(Periodic) Computed and cached diffusion kernel for dt={dt}")

        if method == "fft":
            if not hasattr(self, "_diffusion_kernel_fft"):
                self._diffusion_kernel_fft = np.fft.rfftn(
                    self._diffusion_kernel, s=self.odor_grid.shape
                )
                logger.info(
                    f"(Periodic) Computed and cached FFT of diffusion kernel for dt={dt}"
                )
            odor_fft = np.fft.rfftn(self.odor_grid)
            convolved = np.fft.irfftn(
                odor_fft * self._diffusion_kernel_fft, s=self.odor_grid.shape
            )
            self.odor_grid = convolved
        elif method == "convolve2d":
            self.odor_grid = convolve2d(
                self.odor_grid, self._diffusion_kernel, mode="same", boundary="wrap"
            )
        elif method == "gaussian_filter":
            self.odor_grid = gaussian_filter(
                self.odor_grid, sigma=self._diffusion_sigma, mode="wrap"
            )
        elif method == "box_blur":
            r = int(np.ceil(self._diffusion_sigma * np.sqrt(3)))
            size = 2 * r + 1
            self.odor_grid = uniform_filter(self.odor_grid, size=size, mode="wrap")
            self.odor_grid = uniform_filter(self.odor_grid, size=size, mode="wrap")
            self.odor_grid = uniform_filter(self.odor_grid, size=size, mode="wrap")
        else:
            raise ValueError(
                "Unknown diffusion method: choose 'fft', 'convolve2d', 'gaussian_filter', or 'box_blur'."
            )
        self.odor_grid *= 1 - self.odor_decay_rate
        if hasattr(self, "base_odor"):
            self.odor_grid = np.maximum(self.odor_grid, self.base_odor)


#####################################################################
# Numba-accelerated helper functions for vectorized operations
#####################################################################


@njit
def world_to_grid_periodic_vectorized_numba(
    xs: np.ndarray,
    ys: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated conversion of world coordinates to grid indices and fractional parts.

    Args:
        xs (np.ndarray): 1D NumPy array of x coordinates.
        ys (np.ndarray): 1D NumPy array of y coordinates.
        x_min (float): Minimum x of the grid.
        y_min (float): Minimum y of the grid.
        resolution (float): Grid resolution.
        nx (int): Number of columns in grid.
        ny (int): Number of rows in grid.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of 1D NumPy
            arrays (i, j, fi, fj).
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = (np.floor(gx) % nx).astype(np.int32)
    i = (np.floor(gy) % ny).astype(np.int32)
    fi = gx - np.floor(gx)
    fj = gy - np.floor(gy)
    return i, j, fi, fj


@njit
def get_odor_periodic_vectorized_numba(
    xs: np.ndarray,
    ys: np.ndarray,
    odor_grid: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Numba-accelerated bilinear interpolation for odor values with periodic boundaries.

    Args:
        xs (np.ndarray): 1D NumPy array of x coordinates.
        ys (np.ndarray): 1D NumPy array of y coordinates.
        odor_grid (np.ndarray): 2D NumPy array representing the odor grid.
        x_min (float): Minimum x of the grid.
        y_min (float): Minimum y of the grid.
        resolution (float): Grid resolution.
        nx (int): Number of columns in grid.
        ny (int): Number of rows in grid.

    Returns:
        np.ndarray: 1D NumPy array of interpolated odor values.
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = (np.floor(gx) % nx).astype(np.int32)
    i = (np.floor(gy) % ny).astype(np.int32)
    fi = gx - np.floor(gx)
    fj = gy - np.floor(gy)
    result = np.empty(xs.shape, dtype=odor_grid.dtype)
    for idx in range(xs.size):
        i1 = (i[idx] + 1) % ny
        j1 = (j[idx] + 1) % nx
        Q11 = odor_grid[i[idx], j[idx]]
        Q21 = odor_grid[i[idx], j1]
        Q12 = odor_grid[i1, j[idx]]
        Q22 = odor_grid[i1, j1]
        result[idx] = (
            Q11 * (1 - fi[idx]) * (1 - fj[idx])
            + Q21 * fi[idx] * (1 - fj[idx])
            + Q12 * (1 - fi[idx]) * fj[idx]
            + Q22 * fi[idx] * fj[idx]
        )
    return result


@njit
def is_free_periodic_vectorized_numba(
    xs: np.ndarray,
    ys: np.ndarray,
    wall_mask: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Numba-accelerated check of free space for multiple positions with periodic boundaries.

    Args:
        xs (np.ndarray): 1D array of x coordinates.
        ys (np.ndarray): 1D array of y coordinates.
        wall_mask (np.ndarray): 2D boolean array indicating blocked cells.
        x_min (float): Minimum x of the grid.
        y_min (float): Minimum y of the grid.
        resolution (float): Grid resolution.
        nx (int): Number of columns.
        ny (int): Number of rows.

    Returns:
        np.ndarray: Boolean array (True if free).
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = (np.floor(gx) % nx).astype(np.int32)
    i = (np.floor(gy) % ny).astype(np.int32)
    result = np.empty(xs.shape, dtype=np.bool_)
    for idx in range(xs.size):
        result[idx] = not wall_mask[i[idx], j[idx]]
    return result


@njit
def update_odor_periodic_vectorized_numba(
    odor_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    odors: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Numba-accelerated update of the odor grid with periodic boundaries.

    Args:
        odor_grid (np.ndarray): 2D array representing the odor grid.
        xs (np.ndarray): 1D array of x deposit positions.
        ys (np.ndarray): 1D array of y deposit positions.
        odors (np.ndarray): 1D array of odor values to add.
        x_min (float): Minimum x of the grid.
        y_min (float): Minimum y of the grid.
        resolution (float): Grid resolution.
        nx (int): Number of columns.
        ny (int): Number of rows.

    Returns:
        np.ndarray: The updated odor grid.
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = (np.floor(gx) % nx).astype(np.int32)
    i = (np.floor(gy) % ny).astype(np.int32)
    for idx in range(xs.size):
        odor_grid[i[idx], j[idx]] += odors[idx]
    return odor_grid


@njit
def deposit_odor_kernel_periodic_direct(
    odor_grid: np.ndarray,
    i_center: int,
    j_center: int,
    kernel: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    """
    Numba-accelerated deposition of a single kernel at a grid index with periodic wrapping.

    Args:
        odor_grid (np.ndarray): 2D array of the odor grid.
        i_center (int): Center row index for deposit.
        j_center (int): Center column index for deposit.
        kernel (np.ndarray): 2D array representing the deposition kernel.
        ny (int): Number of rows in odor_grid.
        nx (int): Number of columns in odor_grid.

    Returns:
        np.ndarray: Updated odor_grid.
    """
    ksize = kernel.shape[0]
    half_size = ksize // 2
    for di in range(ksize):
        for dj in range(ksize):
            # Compute target indices with periodic wrapping
            i = (i_center - half_size + di) % ny
            j = (j_center - half_size + dj) % nx
            odor_grid[i, j] += kernel[di, dj]
    return odor_grid


@njit
def deposit_odor_kernels_periodic_vectorized_numba(
    odor_grid: np.ndarray,
    i_centers: np.ndarray,
    j_centers: np.ndarray,
    kernel: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    """
    Numba-accelerated vectorized deposition of the same kernel at multiple grid indices.

    Args:
        odor_grid (np.ndarray): 2D array representing the odor grid.
        i_centers (np.ndarray): 1D array of center row indices.
        j_centers (np.ndarray): 1D array of center column indices.
        kernel (np.ndarray): 2D array representing the kernel.
        ny (int): Number of rows in odor_grid.
        nx (int): Number of columns in odor_grid.

    Returns:
        np.ndarray: Updated odor_grid.
    """
    for idx in range(i_centers.size):
        odor_grid = deposit_odor_kernel_periodic_direct(
            odor_grid, i_centers[idx], j_centers[idx], kernel, ny, nx
        )
    return odor_grid


#####################################################################
# End of Numba helper functions
#####################################################################


#####################################################################
# Derived Arenas
#####################################################################


def create_circular_arena_with_annular_trail(
    config: SimulationConfig,
    arena_radius: float = 75.0,
    trail_radius: float = 50.0,
    trail_width: float = 3.0,
    trail_odor: float = 1.0,
) -> GridArena:
    """Helper to create a circular arena with an annular trail preset in the odor grid."""
    arena = GridArena(
        config.grid_x_min,
        config.grid_x_max,
        config.grid_y_min,
        config.grid_y_max,
        config.grid_resolution,
    )
    # Create coordinate grids.
    y_coords = np.linspace(config.grid_y_min, config.grid_y_max, arena.ny)
    x_coords = np.linspace(config.grid_x_min, config.grid_x_max, arena.nx)
    X, Y = np.meshgrid(x_coords, y_coords)
    distances = np.sqrt(X**2 + Y**2)
    # Set cells outside the circular arena as walls.
    arena.wall_mask = distances > arena_radius
    inner_bound = trail_radius - trail_width / 2
    outer_bound = trail_radius + trail_width / 2
    annulus_mask = (
        (distances >= inner_bound) & (distances <= outer_bound) & (~arena.wall_mask)
    )
    arena.odor_grid[annulus_mask] = trail_odor
    return arena


def create_pbc_arena_with_line(
    config: SimulationConfig, line_width: float = 5.0, line_odor: float = 1.0
) -> PeriodicSquareArena:
    """Helper to create a periodic square arena with a line preset in the odor grid."""
    arena = PeriodicSquareArena(
        config.grid_x_min,
        config.grid_x_max,
        config.grid_y_min,
        config.grid_y_max,
        config.grid_resolution,
        config,
    )
    # Create coordinate grids.
    y_coords = np.linspace(config.grid_y_min, config.grid_y_max, arena.ny)
    x_coords = np.linspace(config.grid_x_min, config.grid_x_max, arena.nx)
    X, Y = np.meshgrid(x_coords, y_coords)
    # Set cells along the line as odor.
    line_mask = np.abs(X) <= line_width / 2
    arena.odor_grid[line_mask] = line_odor
    return arena
