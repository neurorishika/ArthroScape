# arthroscape/sim/visualization.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from typing import List, Dict
from .config import SimulationConfig
from .arena import GridArena
from matplotlib.collections import LineCollection

class VisualizationPipeline:
    def __init__(self, sim_results: List[Dict], config: SimulationConfig, arena: GridArena):
        """
        Initialize the visualization pipeline with simulation results, configuration, and arena.
        :param sim_results: A list of simulation result dictionaries.
        :param config: The SimulationConfig instance.
        :param arena: The GridArena used in the simulation.
        """
        self.sim_results = sim_results
        self.config = config
        self.arena = arena

    def plot_final_odor_grid(self, downsample_factor: int = 1, show: bool = True, save_path: str = None) -> None:
        """
        Plot the final odor grid as a heatmap.
        The downsample_factor parameter allows you to reduce the resolution for visualization.
        """
        # Downsample the grid if needed.
        odor_grid = self.arena.odor_grid
        if downsample_factor > 1:
            odor_grid = odor_grid[::downsample_factor, ::downsample_factor]
            grid_x_min = self.config.grid_x_min
            grid_x_max = self.config.grid_x_max
            grid_y_min = self.config.grid_y_min
            grid_y_max = self.config.grid_y_max
            extent = [grid_x_min, grid_x_max, grid_y_min, grid_y_max]
        else:
            extent = [self.config.grid_x_min, self.config.grid_x_max,
                      self.config.grid_y_min, self.config.grid_y_max]

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(odor_grid, origin='lower', extent=extent, cmap='viridis')
        ax.set_title("Final Odor Grid")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, label="Odor Intensity")
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    def plot_odor_time_series(self, sim_index: int = 0, show: bool = True, save_path: str = None) -> None:
        """
        Plot the time series of odor intensity at the left and right antenna for a simulation replicate.
        """
        result = self.sim_results[sim_index]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(result["odor_left"], label="Left Antenna", lw=2)
        ax.plot(result["odor_right"], label="Right Antenna", lw=2)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Odor Intensity")
        ax.set_title("Odor Time Series at Antennae")
        ax.legend(loc='upper right')
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    def plot_trajectory_with_odor(self, sim_index: int = 0, show: bool = True, save_path: str = None) -> None:
        """
        Plot the fly trajectory with a line colored by the average odor intensity.
        """
        result = self.sim_results[sim_index]
        x = np.array(result["x"])
        y = np.array(result["y"])
        avg_odor = np.array(result["odor_left"]) + np.array(result["odor_right"])
        avg_odor = avg_odor / 2.0

        # Create line segments for colored line
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(vmin=avg_odor.min(), vmax=avg_odor.max()))
        lc.set_array(avg_odor[:-1])
        lc.set_linewidth(2)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.add_collection(lc)
        ax.set_xlim(self.config.grid_x_min, self.config.grid_x_max)
        ax.set_ylim(self.config.grid_y_min, self.config.grid_y_max)
        ax.set_title("Fly Trajectory Colored by Average Odor")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect('equal')
        fig.colorbar(lc, ax=ax, label="Average Odor Intensity")
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    def animate_enhanced_trajectory(self, sim_index: int = 0, interval: int = 50, save_path: str = None) -> None:
        """
        Enhanced animation: animates the fly's trajectory, draws an arrow for heading, and plots the antenna positions.
        The antenna markers are scaled and colored based on the odor intensity.
        """
        result = self.sim_results[sim_index]
        config = self.config
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(config.grid_x_min, config.grid_x_max)
        ax.set_ylim(config.grid_y_min, config.grid_y_max)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Enhanced Fly Trajectory Animation")
        ax.set_aspect('equal')
        
        # Trajectory line (initially empty)
        traj_line, = ax.plot([], [], lw=2, color="black")
        # Create an arrow (will be updated each frame)
        heading_arrow = None
        
        # Create scatter objects for antennas (left and right)
        left_scatter = ax.scatter([], [], s=[], c=[], cmap='viridis', edgecolors='k', label='Left Antenna')
        right_scatter = ax.scatter([], [], s=[], c=[], cmap='viridis', edgecolors='k', label='Right Antenna')
        
        def init():
            traj_line.set_data([], [])
            nonlocal heading_arrow
            if heading_arrow is not None:
                heading_arrow.remove()
            # Initialize antenna scatters with empty 2D arrays
            left_scatter.set_offsets(np.empty((0,2)))
            right_scatter.set_offsets(np.empty((0,2)))
            left_scatter.set_sizes(np.array([]))
            right_scatter.set_sizes(np.array([]))
            left_scatter.set_array(np.array([]))
            right_scatter.set_array(np.array([]))
            return traj_line, left_scatter, right_scatter
        
        def update(frame):
            # Update trajectory
            traj_line.set_data(result["x"][:frame], result["y"][:frame])
            cur_x = result["x"][frame-1]
            cur_y = result["y"][frame-1]
            cur_heading = result["heading"][frame-1]
            
            nonlocal heading_arrow
            if heading_arrow is not None:
                heading_arrow.remove()
            arrow_length = 10  # mm
            dx = arrow_length * math.cos(cur_heading)
            dy = arrow_length * math.sin(cur_heading)
            heading_arrow = ax.arrow(cur_x, cur_y, dx, dy, head_width=2, head_length=4, fc='orange', ec='orange')
            
            # Compute antenna positions using configured offsets.
            left_dx = config.antenna_left_offset[0] * math.cos(cur_heading) - config.antenna_left_offset[1] * math.sin(cur_heading)
            left_dy = config.antenna_left_offset[0] * math.sin(cur_heading) + config.antenna_left_offset[1] * math.cos(cur_heading)
            right_dx = config.antenna_right_offset[0] * math.cos(cur_heading) - config.antenna_right_offset[1] * math.sin(cur_heading)
            right_dy = config.antenna_right_offset[0] * math.sin(cur_heading) + config.antenna_right_offset[1] * math.cos(cur_heading)
            left_x = cur_x + left_dx
            left_y = cur_y + left_dy
            right_x = cur_x + right_dx
            right_y = cur_y + right_dy
            
            # Get odor intensities for antennas (from simulation result)
            odor_left = result["odor_left"][frame-1]
            odor_right = result["odor_right"][frame-1]
            
            # Update scatter positions and marker sizes (scale marker size with odor)
            left_scatter.set_offsets(np.array([[left_x, left_y]]))
            right_scatter.set_offsets(np.array([[right_x, right_y]]))
            base_size = 100  # adjust as needed
            left_scatter.set_sizes(np.array([base_size * odor_left]))
            right_scatter.set_sizes(np.array([base_size * odor_right]))
            left_scatter.set_array(np.array([odor_left]))
            right_scatter.set_array(np.array([odor_right]))
            
            return traj_line, heading_arrow, left_scatter, right_scatter

        ani = animation.FuncAnimation(fig, update, frames=len(result["x"]), init_func=init,
                                      interval=interval, blit=False)
        if save_path:
            ani.save(save_path, writer='imagemagick')
        else:
            plt.show()
        plt.close(fig)


