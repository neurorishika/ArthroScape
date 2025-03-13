# arthroscape/sim/visualizer.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from typing import List, Dict
from .config import SimulationConfig
from .arena import GridArena

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

    def plot_trajectory(self, sim_index: int = 0, show: bool = True, save_path: str = None) -> None:
        """
        Plot the fly trajectory along with arena boundaries and annular trail boundaries.
        """
        result = self.sim_results[sim_index]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(result["x"], result["y"], lw=2, label="Trajectory")
        ax.scatter(result["x"][0], result["y"][0], color='green', label='Start')
        ax.scatter(result["x"][-1], result["y"][-1], color='red', label='End')

        # Draw circular arena boundary.
        arena_boundary = plt.Circle((0, 0), 75, color='black', fill=False, linestyle='--', label='Arena Boundary')
        ax.add_artist(arena_boundary)
        # Draw annular trail boundaries.
        inner_circle = plt.Circle((0, 0), 50 - (3/2), color='blue', fill=False, linestyle='--', label='Trail Boundaries')
        outer_circle = plt.Circle((0, 0), 50 + (3/2), color='blue', fill=False, linestyle='--')
        ax.add_artist(inner_circle)
        ax.add_artist(outer_circle)

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Fly Trajectory")
        ax.set_aspect('equal')
        ax.legend()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)

    def plot_final_odor_grid(self, show: bool = True, save_path: str = None) -> None:
        """
        Plot the final odor grid as a heatmap.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        extent = [self.config.grid_x_min, self.config.grid_x_max, self.config.grid_y_min, self.config.grid_y_max]
        im = ax.imshow(self.arena.odor_grid, origin='lower', extent=extent, cmap='viridis')
        ax.set_title("Final Odor Grid")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, label="Odor Intensity")
        if save_path:
            plt.savefig(save_path)
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
        ax.legend()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)

    def animate_trajectory(self, sim_index: int = 0, interval: int = 50, save_path: str = None) -> None:
        """
        Animate the fly's trajectory over time.
        :param sim_index: Index of the simulation replicate to animate.
        :param interval: Delay between frames in milliseconds.
        :param save_path: If provided, save the animation (e.g., as a GIF).
        """
        result = self.sim_results[sim_index]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(self.config.grid_x_min, self.config.grid_x_max)
        ax.set_ylim(self.config.grid_y_min, self.config.grid_y_max)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Fly Trajectory Animation")
        ax.set_aspect('equal')

        traj_line, = ax.plot([], [], lw=2, color="blue")
        start_point = ax.scatter([], [], color="green", label="Start")
        current_point = ax.scatter([], [], color="red", label="Current")
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            traj_line.set_data([], [])
            start_point.set_offsets([])
            current_point.set_offsets([])
            time_text.set_text("")
            return traj_line, start_point, current_point, time_text

        def update(frame):
            x_data = result["x"][:frame]
            y_data = result["y"][:frame]
            traj_line.set_data(x_data, y_data)
            start_point.set_offsets([result["x"][0], result["y"][0]])
            current_point.set_offsets([result["x"][frame-1], result["y"][frame-1]])
            time_text.set_text(f"Frame: {frame}")
            return traj_line, start_point, current_point, time_text

        ani = animation.FuncAnimation(fig, update, frames=len(result["x"]), init_func=init,
                                      interval=interval, blit=True)

        if save_path:
            ani.save(save_path, writer='imagemagick')
        else:
            plt.show()
        plt.close(fig)

    def animate_odor_grid(self, sim_index: int = 0, interval: int = 200, save_path: str = None) -> None:
        """
        Animate the evolution of the odor grid over time.
        This function assumes that the simulation result includes a key "odor_grid_history"
        which is a list of 2D NumPy arrays capturing the odor grid at different time points.
        :param sim_index: Index of the simulation replicate to animate.
        :param interval: Delay between frames in milliseconds.
        :param save_path: If provided, save the animation (e.g., as a GIF).
        """
        result = self.sim_results[sim_index]
        if "odor_grid_history" not in result:
            raise ValueError("Simulation result does not contain 'odor_grid_history'. Ensure that your simulation engine records the odor grid over time.")

        grid_history = result["odor_grid_history"]
        fig, ax = plt.subplots(figsize=(8, 8))
        extent = [self.config.grid_x_min, self.config.grid_x_max, self.config.grid_y_min, self.config.grid_y_max]
        im = ax.imshow(grid_history[0], origin='lower', extent=extent, cmap='viridis')
        ax.set_title("Dynamic Odor Grid")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, label="Odor Intensity")

        def update(frame):
            im.set_data(grid_history[frame])
            ax.set_title(f"Dynamic Odor Grid\nFrame: {frame}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(grid_history), interval=interval, blit=True)

        if save_path:
            ani.save(save_path, writer='imagemagick')
        else:
            plt.show()
        plt.close(fig)
