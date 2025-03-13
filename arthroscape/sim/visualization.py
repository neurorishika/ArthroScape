# arthroscape/sim/visualization.py
import matplotlib.pyplot as plt
import math
import numpy as np
from .config import SimulationConfig
from .arena import GridArena

def visualize_simulation(sim_result: dict, config: SimulationConfig, arena: GridArena) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Trajectory plot
    axs[0].plot(sim_result["x"], sim_result["y"], lw=0.5, color='red', label='Trajectory')
    # color segments by odor
    odor = np.array(sim_result["odor_left"]) + np.array(sim_result["odor_right"])
    # get segments where odor is greater than 0
    odor_segments = np.where(odor > 0)[0]
    # loop through segments and plot them
    for i in range(len(odor_segments) - 1):
        start = odor_segments[i]
        end = odor_segments[i + 1]
        axs[0].plot(sim_result["x"][start:end], sim_result["y"][start:end], lw=0.5, color='black')
        
    # Start and end points
    axs[0].scatter(sim_result["x"][0], sim_result["y"][0], color='green', label='Start')
    axs[0].scatter(sim_result["x"][-1], sim_result["y"][-1], color='red', label='End')
    axs[0].set_title("Fly Trajectory")
    axs[0].set_xlabel("x (mm)")
    axs[0].set_ylabel("y (mm)")
    axs[0].set_aspect('equal')
    axs[0].legend()
    # Draw circular arena boundary
    circle = plt.Circle((0, 0), 75, color='black', fill=False, linestyle='--')
    axs[0].add_artist(circle)
    # Draw annular trail boundaries (inner and outer)
    inner_circle = plt.Circle((0, 0), 50 - (3/2), color='blue', fill=False, linestyle='--')
    outer_circle = plt.Circle((0, 0), 50 + (3/2), color='blue', fill=False, linestyle='--')
    axs[0].add_artist(inner_circle)
    axs[0].add_artist(outer_circle)
    
    # Odor grid visualization
    extent = [config.grid_x_min, config.grid_x_max, config.grid_y_min, config.grid_y_max]
    im = axs[1].imshow(arena.odor_grid, origin='lower', extent=extent, cmap='viridis')
    axs[1].set_title("Final Odor Grid")
    axs[1].set_xlabel("x (mm)")
    axs[1].set_ylabel("y (mm)")
    axs[1].set_aspect('equal')
    fig.colorbar(im, ax=axs[1], label="Odor Intensity")
    
    plt.tight_layout()
    plt.show()
