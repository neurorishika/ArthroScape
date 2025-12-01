# Customizing the Arena

ArthroScape allows you to simulate agents in various environments. This tutorial explains how to use existing arena types and how to create custom ones.

## Using Built-in Arenas

ArthroScape comes with a few predefined arena types.

### Periodic Square Arena

This is a square arena with periodic boundary conditions (toroidal topology). Agents that exit one side re-enter from the opposite side.

```python
from arthroscape.sim.config import SimulationConfig
from arthroscape.sim.arena import PeriodicSquareArena

config = SimulationConfig()
arena = PeriodicSquareArena(
    x_min=-50, x_max=50,
    y_min=-50, y_max=50,
    resolution=0.1,
    config=config
)
```

### Circular Arena with Annular Trail

This helper function creates a circular arena with a specific odor trail.

```python
from arthroscape.sim.arena import create_circular_arena_with_annular_trail

arena = create_circular_arena_with_annular_trail(
    config,
    arena_radius=100.0,
    trail_radius=50.0,
    trail_width=10.0,
    trail_odor=1.0
)
```

## Creating a Custom Arena

To create a completely new arena, you can inherit from `GridArena` or `Arena`.

### Example: Rectangular Arena with Obstacles

Here is an example of how to create a rectangular arena with a central obstacle.

```python
import numpy as np
from arthroscape.sim.arena import GridArena

class ObstacleArena(GridArena):
    def __init__(self, width, height, obstacle_radius, config):
        # Define dimensions
        x_min, x_max = -width/2, width/2
        y_min, y_max = -height/2, height/2
        resolution = config.grid_resolution
        
        # Create wall mask (True = wall/obstacle)
        ny = int(np.ceil((y_max - y_min) / resolution)) + 1
        nx = int(np.ceil((x_max - x_min) / resolution)) + 1
        wall_mask = np.zeros((ny, nx), dtype=bool)
        
        # Add central circular obstacle
        y_indices, x_indices = np.ogrid[:ny, :nx]
        y_coords = y_min + y_indices * resolution
        x_coords = x_min + x_indices * resolution
        dist_sq = x_coords**2 + y_coords**2
        wall_mask[dist_sq < obstacle_radius**2] = True
        
        # Initialize base class
        super().__init__(x_min, x_max, y_min, y_max, resolution, wall_mask, config)

# Usage
arena = ObstacleArena(width=100, height=100, obstacle_radius=20, config=config)
```

## Modifying the Odor Landscape

You can manually set the initial odor grid of an arena.

```python
# Set a gradient
for i in range(arena.ny):
    arena.odor_grid[i, :] = i / arena.ny
```

Remember that if you use `ConstantOdorRelease`, the odor grid will be updated dynamically during the simulation.
