# arthroscape/sim/arena.py
import numpy as np
import math
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from .config import SimulationConfig

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

class GridArena(Arena):
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                 resolution: float, wall_mask: Optional[np.ndarray] = None):
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

    def is_free(self, x: float, y: float) -> bool:
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return False
        i, j, _, _ = self._world_to_grid(x, y)
        i = np.clip(i, 0, self.ny - 1)
        j = np.clip(j, 0, self.nx - 1)
        return not self.wall_mask[i, j]

    def update_odor(self, x: float, y: float, odor: float) -> None:
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return
        i, j, _, _ = self._world_to_grid(x, y)
        i = np.clip(i, 0, self.ny - 1)
        j = np.clip(j, 0, self.nx - 1)
        self.odor_grid[i, j] += odor
    
    def deposit_odor_kernel(self, x: float, y: float, kernel: np.ndarray) -> None:
        """
        Deposit a kernel (2D array) onto the odor grid centered at (x, y).
        """
        ksize = kernel.shape[0]  # assume square kernel with odd dimensions
        half_size = ksize // 2
        i_center, j_center, _, _ = self._world_to_grid(x, y)
        for di in range(-half_size, half_size + 1):
            for dj in range(-half_size, half_size + 1):
                i = i_center + di
                j = j_center + dj
                if 0 <= i < self.ny and 0 <= j < self.nx:
                    self.odor_grid[i, j] += kernel[di + half_size, dj + half_size]



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
