# tests/test_diffusion.py
import numpy as np
import pytest
from arthroscape.sim.config import SimulationConfig
from arthroscape.sim.arena import GridArena

def test_diffusion_decay():
    # Create a simple config and arena.
    config = SimulationConfig(T=10, fps=1)  # 10 frames for test
    arena = GridArena(config.grid_x_min, config.grid_x_max,
                      config.grid_y_min, config.grid_y_max,
                      config.grid_resolution, config=config)
    # Put a spike in the center.
    center_x = (config.grid_x_min + config.grid_x_max) / 2
    center_y = (config.grid_y_min + config.grid_y_max) / 2
    arena.update_odor(center_x, center_y, 100.0)
    initial_total = arena.odor_grid.sum()
    # Update odor field (simulate one diffusion/decay step).
    arena.update_odor_field()
    # After diffusion, the total odor should be roughly the same (if no decay) or slightly reduced.
    new_total = arena.odor_grid.sum()
    # Check that some diffusion has occurred.
    assert new_total < initial_total

if __name__ == "__main__":
    pytest.main([__file__])