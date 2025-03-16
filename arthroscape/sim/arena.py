# arthroscape/sim/arena.py
import numpy as np
import math
import logging
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from .config import SimulationConfig
from scipy.signal import fftconvolve, convolve2d
from scipy.ndimage import gaussian_filter, uniform_filter

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
                 resolution: float, wall_mask: Optional[np.ndarray] = None, config: Optional[SimulationConfig] = None):
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
        xs and ys must be NumPy arrays of the same shape.
        """
        gx = (xs - self.x_min) / self.resolution
        gy = (ys - self.y_min) / self.resolution
        j = np.floor(gx).astype(int)
        i = np.floor(gy).astype(int)
        fi = gx - j
        fj = gy - i
        # Clip indices to valid range for interpolation
        i = np.clip(i, 0, self.ny - 2)
        j = np.clip(j, 0, self.nx - 2)
        Q11 = self.odor_grid[i, j]
        Q21 = self.odor_grid[i, j+1]
        Q12 = self.odor_grid[i+1, j]
        Q22 = self.odor_grid[i+1, j+1]
        return Q11 * (1 - fi) * (1 - fj) + Q21 * fi * (1 - fj) + Q12 * (1 - fi) * fj + Q22 * fi * fj

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
        Returns a boolean array of the same shape as xs and ys.
        """
        gx = (xs - self.x_min) / self.resolution
        gy = (ys - self.y_min) / self.resolution
        j = np.floor(gx).astype(int)
        i = np.floor(gy).astype(int)
        i = np.clip(i, 0, self.ny - 1)
        j = np.clip(j, 0, self.nx - 1)
        return ~self.wall_mask[i, j]

    def update_odor(self, x: float, y: float, odor: float) -> None:
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return
        i, j, _, _ = self._world_to_grid(x, y)
        i = np.clip(i, 0, self.ny - 1)
        j = np.clip(j, 0, self.nx - 1)
        self.odor_grid[i, j] += odor

    def deposit_odor_kernel(self, x: float, y: float, kernel: np.ndarray) -> None:
        """
        Deposit a kernel (2D array) onto the odor grid centered at (x, y) using vectorized slicing.
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

    def _compute_diffusion_kernel(self, dt: float) -> Tuple[float, np.ndarray]:
        """
        Compute and return the Gaussian diffusion kernel (and sigma in grid units) for the given dt.
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

    def update_odor_field(self, dt: float = 1.0, method: str = 'box_blur') -> None:
        """
        Update the odor grid to simulate diffusion and decay.
        Supports methods: 'fft', 'convolve2d', 'gaussian_filter', and 'box_blur'.
        
        Parameters:
            dt (float): Time step in seconds.
            method (str): Convolution method to use.
        """
        if self.diffusion_coefficient == 0:
            self.odor_grid *= (1 - self.odor_decay_rate)
            if hasattr(self, 'base_odor'):
                self.odor_grid = np.maximum(self.odor_grid, self.base_odor)
            return
        
        if not hasattr(self, '_diffusion_dt'):
            self._diffusion_dt = dt

        # Compute or update the diffusion kernel and sigma if needed.
        if not hasattr(self, '_diffusion_kernel') or self._diffusion_dt != dt:
            sigma_grid, kernel = self._compute_diffusion_kernel(dt)
            self._diffusion_sigma = sigma_grid
            self._diffusion_kernel = kernel
            self._diffusion_dt = dt
            logger.info(f"Computed and cached diffusion kernel for dt={dt}")

        if method == 'fft':
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


def create_circular_arena_with_annular_trail(config: SimulationConfig,
                                             arena_radius: float = 75.0,
                                             trail_radius: float = 50.0,
                                             trail_width: float = 3.0,
                                             trail_odor: float = 1.0) -> GridArena:
    """Helper to create a circular arena with an annular trail preset in the odor grid."""
    arena = GridArena(config.grid_x_min, config.grid_x_max,
                      config.grid_y_min, config.grid_y_max,
                      config.grid_resolution, config=config)
    y_coords = np.linspace(config.grid_y_min, config.grid_y_max, arena.ny)
    x_coords = np.linspace(config.grid_x_min, config.grid_x_max, arena.nx)
    X, Y = np.meshgrid(x_coords, y_coords)
    distances = np.sqrt(X**2 + Y**2)
    arena.wall_mask = distances > arena_radius
    inner_bound = trail_radius - trail_width / 2
    outer_bound = trail_radius + trail_width / 2
    annulus_mask = (distances >= inner_bound) & (distances <= outer_bound) & (~arena.wall_mask)
    arena.odor_grid[annulus_mask] = trail_odor
    arena.base_odor = np.where(annulus_mask, trail_odor, 0.0)
    return arena

# New class for periodic (toroidal) boundaries
class PeriodicSquareArena(GridArena):
    """
    A square arena with periodic (toroidal) boundary conditions.
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                 resolution: float, config: Optional[SimulationConfig] = None):
        # For periodic boundaries, we ignore wall masks.
        super().__init__(
            x_min, x_max, y_min, y_max, resolution,
            wall_mask=np.zeros(
                (int(np.ceil((y_max-y_min)/resolution))+1, int(np.ceil((x_max-x_min)/resolution))+1),
                dtype=bool
            ),
            config=config
        )
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int, float, float]:
        # Wrap coordinates using modulo arithmetic.
        gx = (x - self.x_min) / self.resolution
        gy = (y - self.y_min) / self.resolution
        j = int(np.floor(gx)) % self.nx
        i = int(np.floor(gy)) % self.ny
        fi = gx - np.floor(gx)
        fj = gy - np.floor(gy)
        return i, j, fi, fj

    def get_odor(self, x: float, y: float) -> float:
        i, j, fi, fj = self._world_to_grid(x, y)
        i1 = (i + 1) % self.ny
        j1 = (j + 1) % self.nx
        Q11 = self.odor_grid[i, j]
        Q21 = self.odor_grid[i, j1]
        Q12 = self.odor_grid[i1, j]
        Q22 = self.odor_grid[i1, j1]
        return (Q11 * (1-fi) * (1-fj) +
                Q21 * fi * (1-fj) +
                Q12 * (1-fi) * fj +
                Q22 * fi * fj)

    def is_free(self, x: float, y: float) -> bool:
        # Override to wrap coordinates instead of checking against boundaries.
        i, j, _, _ = self._world_to_grid(x, y)
        return not self.wall_mask[i, j]

    def update_odor(self, x: float, y: float, odor: float) -> None:
        i, j, _, _ = self._world_to_grid(x, y)
        self.odor_grid[i, j] += odor

    def deposit_odor_kernel(self, x: float, y: float, kernel: np.ndarray) -> None:
        """
        Vectorized deposition of a kernel onto the odor grid using np.roll.
        """
        ksize = kernel.shape[0]  # assumed odd
        half_size = ksize // 2
        i_center, j_center, _, _ = self._world_to_grid(x, y)
        deposit_array = np.zeros_like(self.odor_grid)
        deposit_array[:ksize, :ksize] = kernel
        shift_i = i_center - half_size
        shift_j = j_center - half_size
        deposit_array = np.roll(deposit_array, shift=(shift_i, shift_j), axis=(0, 1))
        self.odor_grid += deposit_array

    def update_odor_field(self, dt: float = 1.0, method: str = 'box_blur') -> None:
        # In periodic arenas, use 'wrap' boundary mode.
        if self.diffusion_coefficient == 0:
            self.odor_grid *= (1 - self.odor_decay_rate)
            # Optionally clamp the odor grid if needed.
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
        # Optionally clamp the odor grid if needed.
        if hasattr(self, 'base_odor'):
            self.odor_grid = np.maximum(self.odor_grid, self.base_odor)
        
