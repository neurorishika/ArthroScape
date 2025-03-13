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
        Initialize the visualization pipeline.
        :param sim_results: A list of simulation result dictionaries.
                             For multi-animal simulations, each result has a key "trajectories" (a list of per-animal dicts).
        :param config: The SimulationConfig instance.
        :param arena: The GridArena used in the simulation.
        """
        self.sim_results = sim_results
        self.config = config
        self.arena = arena

    def plot_trajectories_with_odor(self, sim_index: int = 0, show: bool = True, save_path: str = None) -> None:
        """
        Plot all animals' trajectories (each colored by its average odor intensity).
        If only one animal is present, behaves as before.
        """
        result = self.sim_results[sim_index]
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(result["trajectories"])))
        for idx, traj in enumerate(result["trajectories"]):
            x = np.array(traj["x"])
            y = np.array(traj["y"])
            avg_odor = (np.array(traj["odor_left"]) + np.array(traj["odor_right"])) / 2.0
            # Create line segments and a LineCollection
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Use a separate colormap for each animal or simply set the color from our cycle.
            lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(vmin=avg_odor.min(), vmax=avg_odor.max()))
            lc.set_array(avg_odor[:-1])
            lc.set_linewidth(0.5)
            # Optionally, overlay with a solid line in the animal's assigned color.
            ax.add_collection(lc)
            ax.plot(x, y, color=colors[idx], alpha=0.3)
        ax.set_xlim(self.config.grid_x_min, self.config.grid_x_max)
        ax.set_ylim(self.config.grid_y_min, self.config.grid_y_max)
        ax.set_title("Fly Trajectories Colored by Average Odor")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect('equal')
        fig.colorbar(lc, ax=ax, label="Average Odor Intensity")
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    def plot_final_odor_grid(self, downsample_factor: int = 1, show: bool = True, save_path: str = None) -> None:
        """
        Plot the final odor grid as a heatmap.
        """
        odor_grid = self.arena.odor_grid
        if downsample_factor > 1:
            odor_grid = odor_grid[::downsample_factor, ::downsample_factor]
            extent = [self.config.grid_x_min, self.config.grid_x_max,
                      self.config.grid_y_min, self.config.grid_y_max]
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
        Plot time series of odor intensity at the antennae for all animals.
        """
        result = self.sim_results[sim_index]
        fig, ax = plt.subplots(figsize=(10, 4))
        for idx, traj in enumerate(result["trajectories"]):
            ax.plot(traj["odor_left"], label=f"Animal {idx+1} Left", lw=2)
            ax.plot(traj["odor_right"], label=f"Animal {idx+1} Right", lw=2, linestyle="--")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Odor Intensity")
        ax.set_title("Odor Time Series at Antennae")
        ax.legend(loc='upper right')
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    def animate_enhanced_trajectory(self, sim_index: int = 0, interval: int = 50, frame_skip: int = 5, save_path: str = None) -> None:
        """
        Enhanced animation: animate trajectories for all animals with heading arrows and antenna markers.
        Each animal is drawn in a unique color.
        
        :param sim_index: Index of the simulation replicate to animate.
        :param interval: Delay between frames in milliseconds.
        :param frame_skip: Only animate every Nth frame to speed up the animation.
        :param save_path: If provided, save the animation (e.g., as a GIF).
        """
        result = self.sim_results[sim_index]
        cfg = self.config
        num = len(result["trajectories"])
        colors = plt.cm.rainbow(np.linspace(0, 1, num))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(cfg.grid_x_min, cfg.grid_x_max)
        ax.set_ylim(cfg.grid_y_min, cfg.grid_y_max)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Enhanced Multi-Animal Trajectory Animation")
        ax.set_aspect('equal')
        
        # For each animal, store trajectory line, heading arrow, and antenna scatter objects.
        traj_lines = []
        heading_arrows = [None] * num
        left_scatters = []
        right_scatters = []
        for idx in range(num):
            line, = ax.plot([], [], lw=0.5, color=colors[idx])
            traj_lines.append(line)
            ls = ax.scatter([], [], s=[], c=[], cmap='viridis', edgecolors='k')
            rs = ax.scatter([], [], s=[], c=[], cmap='viridis', edgecolors='k')
            left_scatters.append(ls)
            right_scatters.append(rs)
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        # Define the frames to animate by skipping frames.
        total_frames = len(result["trajectories"][0]["x"])
        frames = range(0, total_frames, frame_skip)
        
        def init():
            for line in traj_lines:
                line.set_data([], [])
            for idx in range(num):
                left_scatters[idx].set_offsets(np.empty((0,2)))
                right_scatters[idx].set_offsets(np.empty((0,2)))
                left_scatters[idx].set_sizes(np.array([]))
                right_scatters[idx].set_sizes(np.array([]))
                left_scatters[idx].set_array(np.array([]))
                right_scatters[idx].set_array(np.array([]))
                if heading_arrows[idx] is not None:
                    heading_arrows[idx].remove()
            time_text.set_text("")
            return traj_lines + left_scatters + right_scatters + [time_text]
        
        def update(frame):
            for idx, traj in enumerate(result["trajectories"]):
                x_data = traj["x"][:frame]
                y_data = traj["y"][:frame]
                traj_lines[idx].set_data(x_data, y_data)
                cur_x = traj["x"][frame-1]
                cur_y = traj["y"][frame-1]
                cur_heading = traj["heading"][frame-1]
                # Update heading arrow.
                if heading_arrows[idx] is not None:
                    heading_arrows[idx].remove()
                arrow_length = 1.5  # mm
                dx = arrow_length * math.cos(cur_heading)
                dy = arrow_length * math.sin(cur_heading)
                heading_arrows[idx] = ax.arrow(cur_x, cur_y, dx, dy, head_width=1, head_length=2,
                                                fc=colors[idx], ec=colors[idx])
                # Compute antenna positions.
                left_dx = cfg.antenna_left_offset[0] * math.cos(cur_heading) - cfg.antenna_left_offset[1] * math.sin(cur_heading)
                left_dy = cfg.antenna_left_offset[0] * math.sin(cur_heading) + cfg.antenna_left_offset[1] * math.cos(cur_heading)
                right_dx = cfg.antenna_right_offset[0] * math.cos(cur_heading) - cfg.antenna_right_offset[1] * math.sin(cur_heading)
                right_dy = cfg.antenna_right_offset[0] * math.sin(cur_heading) + cfg.antenna_right_offset[1] * math.cos(cur_heading)
                left_x = cur_x + left_dx
                left_y = cur_y + left_dy
                right_x = cur_x + right_dx
                right_y = cur_y + right_dy
                # Odor intensities.
                odor_left = traj["odor_left"][frame-1]
                odor_right = traj["odor_right"][frame-1]
                # Update antenna scatters.
                left_scatters[idx].set_offsets(np.array([[left_x, left_y]]))
                right_scatters[idx].set_offsets(np.array([[right_x, right_y]]))
                base_size = 100  # adjust as needed
                left_scatters[idx].set_sizes(np.array([base_size * odor_left]))
                right_scatters[idx].set_sizes(np.array([base_size * odor_right]))
                left_scatters[idx].set_array(np.array([odor_left]))
                right_scatters[idx].set_array(np.array([odor_right]))
            time_text.set_text(f"Frame: {frame}")
            return traj_lines + left_scatters + right_scatters + [time_text] + heading_arrows
        
        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                    interval=interval, blit=False)
        if save_path:
            ani.save(save_path, writer='imagemagick')
        else:
            plt.show()
        plt.close(fig)