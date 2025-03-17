# arthroscape/sim/arena.py
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
    @abstractmethod
    def get_odor(self, x: float, y: float) -> float:
        """Return the odor concentration at (x, y) using interpolation."""
        pass

    @abstractmethod
    def is_free(self, x: float, y: float) -> bool:
        """Return True if (x, y) is free (not blocked by a wall)."""
        pass

    @abstractmethod
    def update_odor(self, x: float, y: float, odor: float) -> None:
        """Deposit odor permanently at (x, y) on the grid."""
        pass

    @abstractmethod
    def update_odor_field(self, dt: float = 1.0, method: str = 'gaussian_filter') -> None:
        """Update the odor grid (e.g. diffusion and decay)."""
        pass

class GridArena(Arena):
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                 resolution: float, wall_mask: Optional[np.ndarray] = None,
                 config: Optional[SimulationConfig] = None):
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
        gx = (x - self.x_min) / self.resolution
        gy = (y - self.y_min) / self.resolution
        j = int(np.floor(gx))
        i = int(np.floor(gy))
        fi = gx - j
        fj = gy - i
        return i, j, fi, fj

    def world_to_grid_vectorized(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert arrays of world coordinates to grid indices and fractional parts.
        
        :param x: NumPy array of x coordinates.
        :param y: NumPy array of y coordinates.
        :return: Tuple of arrays (i, j, fi, fj) where i and j are integer grid indices,
                 and fi and fj are the fractional parts.
        """
        return world_to_grid_vectorized_numba(x, y, self.x_min, self.y_min, self.resolution, self.nx, self.ny)


    def get_odor(self, x: float, y: float) -> float:
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return 0.0
        i, j, fi, fj = self._world_to_grid(x, y)
        i1 = min(i, self.ny - 2)
        j1 = min(j, self.nx - 2)
        Q11 = self.odor_grid[i1, j1]
        Q21 = self.odor_grid[i1, j1+1]
        Q12 = self.odor_grid[i1+1, j1]
        Q22 = self.odor_grid[i1+1, j1+1]
        return (Q11 * (1-fi) * (1-fj) +
                Q21 * fi * (1-fj) +
                Q12 * (1-fi) * fj +
                Q22 * fi * fj)
    
    def get_odor_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Vectorized version of get_odor using bilinear interpolation.
        
        :param xs: NumPy array of x coordinates.
        :param ys: NumPy array of y coordinates.
        :return: NumPy array of interpolated odor values.
        """
        return get_odor_vectorized_numba(xs, ys, self.odor_grid, self.x_min, self.y_min, self.resolution, self.nx, self.ny)

    def is_free(self, x: float, y: float) -> bool:
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return False
        i, j, _, _ = self._world_to_grid(x, y)
        i = np.clip(i, 0, self.ny - 1)
        j = np.clip(j, 0, self.nx - 1)
        return not self.wall_mask[i, j]
    
    def is_free_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Vectorized version of is_free.
        
        :param xs: NumPy array of x coordinates.
        :param ys: NumPy array of y coordinates.
        :return: Boolean NumPy array indicating free (True) or blocked (False) for each coordinate.
        """
        return is_free_vectorized_numba(xs, ys, self.wall_mask, self.x_min, self.y_min, self.resolution, self.nx, self.ny)

    def update_odor(self, x: float, y: float, odor: float) -> None:
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return
        i, j, _, _ = self._world_to_grid(x, y)
        i = np.clip(i, 0, self.ny - 1)
        j = np.clip(j, 0, self.nx - 1)
        self.odor_grid[i, j] += odor

    def update_odor_vectorized(self, xs: np.ndarray, ys: np.ndarray, odors: np.ndarray) -> None:
        """
        Vectorized update: deposit odor values at multiple positions.
        
        :param xs: 1D NumPy array of x coordinates.
        :param ys: 1D NumPy array of y coordinates.
        :param odors: 1D NumPy array of odor values.
        """
        self.odor_grid = update_odor_vectorized_numba(self.odor_grid, xs, ys, odors, self.x_min, self.y_min, self.resolution, self.nx, self.ny)

    def deposit_odor_kernel(self, x: float, y: float, kernel: np.ndarray) -> None:
        """
        Deposit a single kernel (2D array) onto the odor grid at a given position.
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
    
    def deposit_odor_kernels_vectorized(self, xs: np.ndarray, ys: np.ndarray, kernel: np.ndarray) -> None:
        """
        Vectorized deposition: deposit the same kernel at multiple positions.
        
        This method uses a vectorized world-to-grid conversion and then loops over the deposit positions.
        Note: Further vectorization would require advanced indexing and careful boundary handling.
        
        :param xs: 1D NumPy array of x coordinates for deposit centers.
        :param ys: 1D NumPy array of y coordinates for deposit centers.
        :param kernel: 2D NumPy array representing the kernel (assumed square, odd dimensions).
        """
        self.odor_grid = deposit_odor_kernels_vectorized_numba(self.odor_grid, xs, ys, kernel, self.ny, self.nx)
    
    def _compute_diffusion_kernel(self, dt: float) -> Tuple[float, np.ndarray]:
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

    def update_odor_field(self, dt: float = 1.0, method: str = 'box_blur') -> None:
        if self.diffusion_coefficient == 0:
            self.odor_grid *= (1 - self.odor_decay_rate)
            if hasattr(self, 'base_odor'):
                self.odor_grid = np.maximum(self.odor_grid, self.base_odor)
            return
        
        if not hasattr(self, '_diffusion_dt'):
            self._diffusion_dt = dt

        if not hasattr(self, '_diffusion_kernel') or self._diffusion_dt != dt:
            sigma_grid, kernel = self._compute_diffusion_kernel(dt)
            self._diffusion_sigma = sigma_grid
            self._diffusion_kernel = kernel
            self._diffusion_dt = dt
            logger.info(f"Computed and cached diffusion kernel for dt={dt}")

        if method == 'fft':
            # FFT-based convolution
            if not hasattr(self, '_diffusion_kernel_fft'):
                self._diffusion_kernel_fft = np.fft.rfftn(self._diffusion_kernel, s=self.odor_grid.shape)
                logger.info("Computed and cached FFT of diffusion kernel.")
            odor_fft = np.fft.rfftn(self.odor_grid)
            convolved = np.fft.irfftn(odor_fft * self._diffusion_kernel_fft, s=self.odor_grid.shape)
            self.odor_grid = convolved
        elif method == 'convolve2d':
            self.odor_grid = convolve2d(self.odor_grid, self._diffusion_kernel, mode='same', boundary='fill', fillvalue=0)
        elif method == 'gaussian_filter':
            self.odor_grid = gaussian_filter(self.odor_grid, sigma=self._diffusion_sigma, mode='constant', cval=0)
        elif method == 'box_blur':
            # Use uniform_filter to approximate Gaussian blur by applying it three times.
            # For a 1D box filter of width w, the variance is (w^2 - 1) / 12.
            # Here we set the box filter width such that its variance approximates that of our Gaussian.
            # A common heuristic is r = ceil(sigma * sqrt(3)), so kernel size = 2*r+1.
            r = int(np.ceil(self._diffusion_sigma * np.sqrt(3)))
            size = 2 * r + 1
            # Apply the box filter three times for a better approximation.
            self.odor_grid = uniform_filter(self.odor_grid, size=size, mode='constant', cval=0)
            self.odor_grid = uniform_filter(self.odor_grid, size=size, mode='constant', cval=0)
            self.odor_grid = uniform_filter(self.odor_grid, size=size, mode='constant', cval=0)
        else:
            raise ValueError("Unknown diffusion method: choose 'fft', 'convolve2d', 'gaussian_filter', or 'box_blur'.")

        # Finally, apply odor decay.
        self.odor_grid *= (1 - self.odor_decay_rate)

        # Optionally clamp the odor grid if needed.
        if hasattr(self, 'base_odor'):
            self.odor_grid = np.maximum(self.odor_grid, self.base_odor)

#####################################################################
# Numba-accelerated helper functions for vectorized operations
#####################################################################

@njit
def world_to_grid_vectorized_numba(xs, ys, x_min, y_min, resolution, nx, ny):
    """
    Numba-accelerated conversion of world coordinates to grid indices and fractional parts.
    
    :param xs: 1D NumPy array of x coordinates.
    :param ys: 1D NumPy array of y coordinates.
    :param x_min: Minimum x of the grid.
    :param y_min: Minimum y of the grid.
    :param resolution: Grid resolution.
    :param nx: Number of columns in grid.
    :param ny: Number of rows in grid.
    :return: Tuple of 1D NumPy arrays (i, j, fi, fj).
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = np.floor(gx).astype(np.int32)
    i = np.floor(gy).astype(np.int32)
    fi = gx - j
    fj = gy - np.floor(gy)
    return i, j, fi, fj

@njit
def get_odor_vectorized_numba(xs, ys, odor_grid, x_min, y_min, resolution, nx, ny):
    """
    Numba-accelerated bilinear interpolation for odor values.
    
    :param xs: 1D NumPy array of x coordinates.
    :param ys: 1D NumPy array of y coordinates.
    :param odor_grid: 2D NumPy array representing the odor grid.
    :param x_min: Minimum x of the grid.
    :param y_min: Minimum y of the grid.
    :param resolution: Grid resolution.
    :param nx: Number of columns in grid.
    :param ny: Number of rows in grid.
    :return: 1D NumPy array of interpolated odor values.
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
        Q21 = odor_grid[i[idx], j[idx]+1]
        Q12 = odor_grid[i[idx]+1, j[idx]]
        Q22 = odor_grid[i[idx]+1, j[idx]+1]
        result[idx] = (Q11 * (1 - fi[idx]) * (1 - fj[idx]) +
                       Q21 * fi[idx] * (1 - fj[idx]) +
                       Q12 * (1 - fi[idx]) * fj[idx] +
                       Q22 * fi[idx] * fj[idx])
    return result

@njit
def is_free_vectorized_numba(xs, ys, wall_mask, x_min, y_min, resolution, nx, ny):
    """
    Numba-accelerated check of free space for multiple positions.
    
    :param xs: 1D array of x coordinates.
    :param ys: 1D array of y coordinates.
    :param wall_mask: 2D boolean array indicating blocked cells.
    :param x_min: Minimum x of the grid.
    :param y_min: Minimum y of the grid.
    :param resolution: Grid resolution.
    :param nx: Number of columns.
    :param ny: Number of rows.
    :return: Boolean array (True if free).
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
def update_odor_vectorized_numba(odor_grid, xs, ys, odors, x_min, y_min, resolution, nx, ny):
    """
    Numba-accelerated update of the odor grid given multiple deposit positions.
    
    This function uses np.add.at to handle repeated indices.
    
    :param odor_grid: 2D array representing the odor grid.
    :param xs: 1D array of x deposit positions.
    :param ys: 1D array of y deposit positions.
    :param odors: 1D array of odor values to add.
    :param x_min: Minimum x of the grid.
    :param y_min: Minimum y of the grid.
    :param resolution: Grid resolution.
    :param nx: Number of columns.
    :param ny: Number of rows.
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
def deposit_odor_kernel_numba(odor_grid, i_center, j_center, kernel, ny, nx):
    """
    Numba-accelerated deposition of a single kernel at a grid index with periodic wrapping.
    
    Parameters:
      odor_grid: 2D array of the odor grid.
      i_center: Center row index (int) for deposit.
      j_center: Center column index (int) for deposit.
      kernel: 2D array representing the deposition kernel.
      ny: Number of rows in odor_grid.
      nx: Number of columns in odor_grid.
    
    Returns:
      Updated odor_grid.
    """
    ksize = kernel.shape[0]
    half_size = ksize // 2

    # type cast to int
    i_center = int(i_center)
    j_center = int(j_center)

    # Compute deposition boundaries on the grid.
    i0 = max(i_center - half_size, 0)
    i1 = min(i_center + half_size + 1, ny)
    j0 = max(j_center - half_size, 0)
    j1 = min(j_center + half_size + 1, nx)

    # Compute corresponding kernel indices.
    ki0 = half_size - (i_center - i0)
    kj0 = half_size - (j_center - j0)

    for di in range(i1 - i0):
        for dj in range(j1 - j0):
            # Explicitly cast kernel indices to int.
            odor_grid[i0 + di, j0 + dj] += kernel[int(ki0) + di, int(kj0) + dj]
    return odor_grid

@njit
def deposit_odor_kernels_vectorized_numba(odor_grid, i_centers, j_centers, kernel, ny, nx):
    """
    Numba-accelerated vectorized deposition of the same kernel at multiple grid indices.
    
    Parameters:
      odor_grid: 2D array representing the odor grid.
      i_centers, j_centers: 1D arrays of center row and column indices (ints).
      kernel: 2D kernel array (assumed square, odd dimensions).
      ny, nx: Grid dimensions.
    
    Returns:
      Updated odor_grid.
    """
    for idx in range(i_centers.size):
        odor_grid = deposit_odor_kernel_numba(odor_grid, i_centers[idx], j_centers[idx], kernel, ny, nx)
    return odor_grid

#####################################################################
# End of Numba helper functions
#####################################################################

class PeriodicSquareArena(GridArena):
    """
    A square arena with periodic (toroidal) boundary conditions.
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                 resolution: float, config: Optional[SimulationConfig] = None):
        super().__init__(
            x_min, x_max, y_min, y_max, resolution,
            wall_mask=np.zeros(
                (int(np.ceil((y_max - y_min) / resolution)) + 1,
                 int(np.ceil((x_max - x_min) / resolution)) + 1),
                dtype=bool
            ),
            config=config
        )
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int, float, float]:
        gx = (x - self.x_min) / self.resolution
        gy = (y - self.y_min) / self.resolution
        j = int(np.floor(gx)) % self.nx
        i = int(np.floor(gy)) % self.ny
        fi = gx - np.floor(gx)
        fj = gy - np.floor(gy)
        return i, j, fi, fj

    def world_to_grid_vectorized(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert arrays of world coordinates to grid indices and fractional parts.
        
        :param x: NumPy array of x coordinates.
        :param y: NumPy array of y coordinates.
        :return: Tuple of arrays (i, j, fi, fj) where i and j are integer grid indices,
                 and fi and fj are the fractional parts.
        """
        return world_to_grid_periodic_vectorized_numba(x, y, self.x_min, self.y_min, self.resolution, self.nx, self.ny)

    def get_odor(self, x: float, y: float) -> float:
        i, j, fi, fj = self._world_to_grid(x, y)
        i1 = (i + 1) % self.ny
        j1 = (j + 1) % self.nx
        Q11 = self.odor_grid[i, j]
        Q21 = self.odor_grid[i, j1]
        Q12 = self.odor_grid[i1, j]
        Q22 = self.odor_grid[i1, j1]
        return (Q11 * (1 - fi) * (1 - fj) +
                Q21 * fi * (1 - fj) +
                Q12 * (1 - fi) * fj +
                Q22 * fi * fj)

    def get_odor_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Vectorized version of get_odor using bilinear interpolation.
        
        :param xs: NumPy array of x coordinates.
        :param ys: NumPy array of y coordinates.
        :return: NumPy array of interpolated odor values.
        """
        return get_odor_periodic_vectorized_numba(xs, ys, self.odor_grid, self.x_min, self.y_min, self.resolution, self.nx, self.ny)

    def is_free(self, x: float, y: float) -> bool:
        i, j, _, _ = self._world_to_grid(x, y)
        return not self.wall_mask[i, j]

    def is_free_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Vectorized version of is_free.
        
        :param xs: NumPy array of x coordinates.
        :param ys: NumPy array of y coordinates.
        :return: Boolean NumPy array indicating free (True) or blocked (False) for each coordinate.
        """
        return is_free_periodic_vectorized_numba(xs, ys, self.wall_mask, self.x_min, self.y_min, self.resolution, self.nx, self.ny)

    def update_odor(self, x: float, y: float, odor: float) -> None:
        i, j, _, _ = self._world_to_grid(x, y)
        self.odor_grid[i, j] += odor

    def update_odor_vectorized(self, xs: np.ndarray, ys: np.ndarray, odors: np.ndarray) -> None:
        """
        Vectorized update: deposit odor values at multiple positions.
        
        :param xs: 1D NumPy array of x coordinates.
        :param ys: 1D NumPy array of y coordinates.
        :param odors: 1D NumPy array of odor values.
        """
        self.odor_grid = update_odor_periodic_vectorized_numba(self.odor_grid, xs, ys, odors, self.x_min, self.y_min, self.resolution, self.nx, self.ny)

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

    def deposit_odor_kernels_vectorized(self, xs: np.ndarray, ys: np.ndarray, kernel: np.ndarray) -> None:
        """
        Vectorized deposition: deposit the same kernel at multiple positions.
        
        This method uses a vectorized world-to-grid conversion and then loops over the deposit positions.
        Note: Further vectorization would require advanced indexing and careful boundary handling.
        
        :param xs: 1D NumPy array of x coordinates for deposit centers.
        :param ys: 1D NumPy array of y coordinates for deposit centers.
        :param kernel: 2D NumPy array representing the kernel (assumed square, odd dimensions).
        """
        self.odor_grid = deposit_odor_kernels_periodic_vectorized_numba(self.odor_grid, xs, ys, kernel, self.ny, self.nx)

    def update_odor_field(self, dt: float = 1.0, method: str = 'box_blur') -> None:
        if self.diffusion_coefficient == 0:
            self.odor_grid *= (1 - self.odor_decay_rate)
            if hasattr(self, 'base_odor'):
                self.odor_grid = np.maximum(self.odor_grid, self.base_odor)
            return
        
        if not hasattr(self, '_diffusion_dt'):
            self._diffusion_dt = dt

        if not hasattr(self, '_diffusion_kernel') or self._diffusion_dt != dt:
            sigma, kernel = self._compute_diffusion_kernel(dt)
            self._diffusion_sigma = sigma
            self._diffusion_kernel = kernel
            self._diffusion_dt = dt
            logger.info(f"(Periodic) Computed and cached diffusion kernel for dt={dt}")

        if method == 'fft':
            if not hasattr(self, '_diffusion_kernel_fft'):
                self._diffusion_kernel_fft = np.fft.rfftn(self._diffusion_kernel, s=self.odor_grid.shape)
                logger.info(f"(Periodic) Computed and cached FFT of diffusion kernel for dt={dt}")
            odor_fft = np.fft.rfftn(self.odor_grid)
            convolved = np.fft.irfftn(odor_fft * self._diffusion_kernel_fft, s=self.odor_grid.shape)
            self.odor_grid = convolved
        elif method == 'convolve2d':
            self.odor_grid = convolve2d(self.odor_grid, self._diffusion_kernel, mode='same', boundary='wrap')
        elif method == 'gaussian_filter':
            self.odor_grid = gaussian_filter(self.odor_grid, sigma=self._diffusion_sigma, mode='wrap')
        elif method == 'box_blur':
            r = int(np.ceil(self._diffusion_sigma * np.sqrt(3)))
            size = 2 * r + 1
            self.odor_grid = uniform_filter(self.odor_grid, size=size, mode='wrap')
            self.odor_grid = uniform_filter(self.odor_grid, size=size, mode='wrap')
            self.odor_grid = uniform_filter(self.odor_grid, size=size, mode='wrap')
        else:
            raise ValueError("Unknown diffusion method: choose 'fft', 'convolve2d', 'gaussian_filter', or 'box_blur'.")
        self.odor_grid *= (1 - self.odor_decay_rate)
        if hasattr(self, 'base_odor'):
            self.odor_grid = np.maximum(self.odor_grid, self.base_odor)

#####################################################################
# Numba-accelerated helper functions for vectorized operations
#####################################################################

@njit
def world_to_grid_periodic_vectorized_numba(xs, ys, x_min, y_min, resolution, nx, ny):
    """
    Numba-accelerated conversion of world coordinates to grid indices and fractional parts.
    
    :param xs: 1D NumPy array of x coordinates.
    :param ys: 1D NumPy array of y coordinates.
    :param x_min: Minimum x of the grid.
    :param y_min: Minimum y of the grid.
    :param resolution: Grid resolution.
    :param nx: Number of columns in grid.
    :param ny: Number of rows in grid.
    :return: Tuple of 1D NumPy arrays (i, j, fi, fj).
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = (np.floor(gx) % nx).astype(np.int32)
    i = (np.floor(gy) % ny).astype(np.int32)
    fi = gx - np.floor(gx)
    fj = gy - np.floor(gy)
    return i, j, fi, fj

@njit
def get_odor_periodic_vectorized_numba(xs, ys, odor_grid, x_min, y_min, resolution, nx, ny):
    """
    Numba-accelerated bilinear interpolation for odor values.
    
    :param xs: 1D NumPy array of x coordinates.
    :param ys: 1D NumPy array of y coordinates.
    :param odor_grid: 2D NumPy array representing the odor grid.
    :param x_min: Minimum x of the grid.
    :param y_min: Minimum y of the grid.
    :param resolution: Grid resolution.
    :param nx: Number of columns in grid.
    :param ny: Number of rows in grid.
    :return: 1D NumPy array of interpolated odor values.
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
        result[idx] = (Q11 * (1 - fi[idx]) * (1 - fj[idx]) +
                       Q21 * fi[idx] * (1 - fj[idx]) +
                       Q12 * (1 - fi[idx]) * fj[idx] +
                       Q22 * fi[idx] * fj[idx])
    return result

@njit
def is_free_periodic_vectorized_numba(xs, ys, wall_mask, x_min, y_min, resolution, nx, ny):
    """
    Numba-accelerated check of free space for multiple positions.
    
    :param xs: 1D array of x coordinates.
    :param ys: 1D array of y coordinates.
    :param wall_mask: 2D boolean array indicating blocked cells.
    :param x_min: Minimum x of the grid.
    :param y_min: Minimum y of the grid.
    :param resolution: Grid resolution.
    :param nx: Number of columns.
    :param ny: Number of rows.
    :return: Boolean array (True if free).
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
def update_odor_periodic_vectorized_numba(odor_grid, xs, ys, odors, x_min, y_min, resolution, nx, ny):
    """
    Numba-accelerated update of the odor grid given multiple deposit positions.
    
    This function uses np.add.at to handle repeated indices.
    
    :param odor_grid: 2D array representing the odor grid.
    :param xs: 1D array of x deposit positions.
    :param ys: 1D array of y deposit positions.
    :param odors: 1D array of odor values to add.
    :param x_min: Minimum x of the grid.
    :param y_min: Minimum y of the grid.
    :param resolution: Grid resolution.
    :param nx: Number of columns.
    :param ny: Number of rows.
    """
    gx = (xs - x_min) / resolution
    gy = (ys - y_min) / resolution
    j = (np.floor(gx) % nx).astype(np.int32)
    i = (np.floor(gy) % ny).astype(np.int32)
    for idx in range(xs.size):
        odor_grid[i[idx], j[idx]] += odors[idx]
    return odor_grid

@njit
def deposit_odor_kernel_periodic_direct(odor_grid, i_center, j_center, kernel):
    """
    Numba-accelerated deposition of a single kernel at a grid index with periodic wrapping.
    """
    # type cast to int
    i_center = int(i_center)
    j_center = int(j_center)
    ny, nx = odor_grid.shape
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
def deposit_odor_kernels_periodic_vectorized_numba(odor_grid, i_centers, j_centers, kernel, ny, nx):
    """
    Numba-accelerated vectorized deposition of the same kernel at multiple grid indices.
    
    :param odor_grid: 2D array representing the odor grid.
    :param i_centers: 1D array of center row indices.
    :param j_centers: 1D array of center column indices.
    :param kernel: 2D array representing the kernel.
    :param ny: Number of rows in odor_grid.
    :param nx: Number of columns in odor_grid.
    """
    for idx in range(i_centers.size):
        odor_grid = deposit_odor_kernel_periodic_direct(odor_grid, i_centers[idx], j_centers[idx], kernel)
    return odor_grid

#####################################################################
# End of Numba helper functions
#####################################################################
    


#####################################################################
# Derived Arenas
#####################################################################

def create_circular_arena_with_annular_trail(config: SimulationConfig,
                                             arena_radius: float = 75.0,
                                             trail_radius: float = 50.0,
                                             trail_width: float = 3.0,
                                             trail_odor: float = 1.0) -> GridArena:
    """Helper to create a circular arena with an annular trail preset in the odor grid."""
    arena = GridArena(config.grid_x_min, config.grid_x_max,
                      config.grid_y_min, config.grid_y_max,
                      config.grid_resolution)
    # Create coordinate grids.
    y_coords = np.linspace(config.grid_y_min, config.grid_y_max, arena.ny)
    x_coords = np.linspace(config.grid_x_min, config.grid_x_max, arena.nx)
    X, Y = np.meshgrid(x_coords, y_coords)
    distances = np.sqrt(X**2 + Y**2)
    # Set cells outside the circular arena as walls.
    arena.wall_mask = distances > arena_radius
    inner_bound = trail_radius - trail_width / 2
    outer_bound = trail_radius + trail_width / 2
    annulus_mask = (distances >= inner_bound) & (distances <= outer_bound) & (~arena.wall_mask)
    arena.odor_grid[annulus_mask] = trail_odor
    return arena