# arthroscape/sim/visualization.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from typing import List, Dict, Any, Tuple
from .config import SimulationConfig
from .arena import GridArena
from matplotlib.collections import LineCollection
import logging

logger = logging.getLogger(__name__)

def wrap_coordinates(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Wrap an array of coordinates into the interval [min_val, max_val)."""
    width = max_val - min_val
    return ((arr - min_val) % width) + min_val

def segment_trajectory_with_indices(x: np.ndarray, y: np.ndarray,
                                    x_min: float, x_max: float,
                                    y_min: float, y_max: float,
                                    threshold: Tuple[float, float] = None) -> List[Tuple[int, int]]:
    """
    Segment a trajectory (x, y) into continuous pieces by detecting jumps.
    
    :param x: 1D array of x coordinates.
    :param y: 1D array of y coordinates.
    :param x_min: Minimum x value.
    :param x_max: Maximum x value.
    :param y_min: Minimum y value.
    :param y_max: Maximum y value.
    :param threshold: Tuple (tx, ty) thresholds; if None, defaults to half domain width and height.
    :return: List of tuples (start_idx, end_idx) for each continuous segment.
    """
    if threshold is None:
        threshold = ((x_max - x_min) / 2.0, (y_max - y_min) / 2.0)
    thresh_x, thresh_y = threshold

    segments = []
    start = 0
    for i in range(1, len(x)):
        if abs(x[i] - x[i-1]) > thresh_x or abs(y[i] - y[i-1]) > thresh_y:
            segments.append((start, i))
            start = i
    segments.append((start, len(x)))
    return segments

class VisualizationPipeline:
    def __init__(self, sim_results: List[Dict[str, Any]], config: SimulationConfig, arena: GridArena):
        """
        Initialize the visualization pipeline.
        
        :param sim_results: A list of simulation result dictionaries.
                            For multi-animal simulations, each result should have a key "trajectories"
                            containing a list of per-animal dictionaries.
        :param config: The SimulationConfig instance.
        :param arena: The GridArena used in the simulation.
        """
        self.sim_results = sim_results
        self.config = config
        self.arena = arena

    def plot_trajectories_with_odor(self, sim_index: int = 0, show: bool = True, 
                                save_path: str = None, wraparound: bool = False) -> None:
        """
        Plot all animals' trajectories colored by average odor intensity.
        
        If wraparound is True, the trajectory is segmented to avoid spurious crossing lines.
        """
        result = self.sim_results[sim_index]
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(result["trajectories"])))
        
        # Iterate over each animal's trajectory.
        for idx, traj in enumerate(result["trajectories"]):
            x = np.array(traj["x"])
            y = np.array(traj["y"])
            
            # If wraparound is enabled, wrap the coordinates and segment the trajectory.
            if wraparound:
                x_wrapped = wrap_coordinates(x, self.config.grid_x_min, self.config.grid_x_max)
                y_wrapped = wrap_coordinates(y, self.config.grid_y_min, self.config.grid_y_max)
                segments_idx = segment_trajectory_with_indices(x_wrapped, y_wrapped,
                                                            self.config.grid_x_min, self.config.grid_x_max,
                                                            self.config.grid_y_min, self.config.grid_y_max)
            else:
                # If not wrapping, treat the whole trajectory as one segment.
                segments_idx = [(0, len(x))]
                x_wrapped = x
                y_wrapped = y

            # Average odor intensity over the trajectory.
            avg_odor = (np.array(traj["odor_left"]) + np.array(traj["odor_right"])) / 2.0

            # Plot each continuous segment separately.
            for (start, end) in segments_idx:
                if end - start < 2:
                    continue  # Skip very short segments
                segment = np.column_stack((x_wrapped[start:end], y_wrapped[start:end]))
                lc = LineCollection([segment], cmap='viridis', 
                                    norm=plt.Normalize(vmin=avg_odor.min(), vmax=avg_odor.max()))
                lc.set_array(avg_odor[start:end-1])
                lc.set_linewidth(0.5)
                ax.add_collection(lc)
            
            # Optionally, plot the entire trajectory with a faint solid line for context.
            # ax.plot(x_wrapped, y_wrapped, color=colors[idx], alpha=0.3)
        
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


    def plot_final_odor_grid(self, downsample_factor: int = 1, show: bool = True,
                             save_path: str = None) -> None:
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

    def plot_odor_time_series(self, sim_index: int = 0, show: bool = True,
                               save_path: str = None) -> None:
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

    def animate_enhanced_trajectory(self, sim_index: int = 0, interval: int = 50, frame_skip: int = 5,
                                    save_path: str = None, wraparound: bool = False) -> None:
        """
        Enhanced animation: animate trajectories for all animals with heading arrows and antenna markers.
        When wraparound is True, trajectories are pre-segmented to avoid spurious connecting lines.
        
        :param sim_index: Index of the simulation replicate.
        :param interval: Delay between frames in milliseconds.
        :param frame_skip: Only animate every Nth frame to speed up the animation.
        :param save_path: Optional file path to save the animation (e.g., as a GIF).
        :param wraparound: If True, pre-segment and wrap coordinates.
        """
        result = self.sim_results[sim_index]
        cfg = self.config
        num = len(result["trajectories"])
        colors = plt.cm.rainbow(np.linspace(0, 1, num))
        
        # Pre-compute segmentation indices for each animal if wraparound is enabled.
        seg_indices = []
        if wraparound:
            for traj in result["trajectories"]:
                x = np.array(traj["x"])
                y = np.array(traj["y"])
                # Wrap the entire trajectory.
                x_wrapped = wrap_coordinates(x, cfg.grid_x_min, cfg.grid_x_max)
                y_wrapped = wrap_coordinates(y, cfg.grid_y_min, cfg.grid_y_max)
                seg_idx = segment_trajectory_with_indices(x_wrapped, y_wrapped,
                                                          cfg.grid_x_min, cfg.grid_x_max,
                                                          cfg.grid_y_min, cfg.grid_y_max)
                seg_indices.append(seg_idx)
        else:
            seg_indices = [ [(0, len(traj["x"]))] for traj in result["trajectories"] ]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(cfg.grid_x_min, cfg.grid_x_max)
        ax.set_ylim(cfg.grid_y_min, cfg.grid_y_max)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Enhanced Multi-Animal Trajectory Animation")
        ax.set_aspect('equal')
        
        # For each animal, create an empty LineCollection for its segments.
        line_collections = []
        for idx in range(num):
            lc = LineCollection([], colors=[colors[idx]], linewidths=0.5)
            line_collections.append(lc)
            ax.add_collection(lc)
        
        # Also create objects for heading arrows and antenna scatters.
        heading_arrows = [None] * num
        left_scatters = []
        right_scatters = []
        for idx in range(num):
            ls = ax.scatter([], [], s=[], c=[], cmap='viridis', edgecolors='k')
            rs = ax.scatter([], [], s=[], c=[], cmap='viridis', edgecolors='k')
            left_scatters.append(ls)
            right_scatters.append(rs)
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        total_frames = len(result["trajectories"][0]["x"])
        frames = range(0, total_frames, frame_skip)
        
        def init():
            for lc in line_collections:
                lc.set_segments([])
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
            return line_collections + left_scatters + right_scatters + [time_text]
        
        def update(frame):
            for idx, traj in enumerate(result["trajectories"]):
                # Get trajectory arrays.
                x = np.array(traj["x"])
                y = np.array(traj["y"])
                # If wraparound, wrap coordinates.
                if wraparound:
                    x = wrap_coordinates(x, cfg.grid_x_min, cfg.grid_x_max)
                    y = wrap_coordinates(y, cfg.grid_y_min, cfg.grid_y_max)
                # Get segmentation indices for this animal.
                segs = seg_indices[idx]
                segments_to_plot = []
                for (start, end) in segs:
                    if frame > start:
                        seg_end = min(end, frame)
                        if seg_end - start > 1:
                            segments_to_plot.append(np.column_stack((x[start:seg_end], y[start:seg_end])))
                line_collections[idx].set_segments(segments_to_plot)
                
                # For heading arrow and antenna markers, use the last point up to current frame.
                cur_idx = frame - 1
                cur_x = x[cur_idx]
                cur_y = y[cur_idx]
                cur_heading = traj["heading"][cur_idx]
                # Remove previous arrow if exists.
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
                odor_left = traj["odor_left"][cur_idx]
                odor_right = traj["odor_right"][cur_idx]
                left_scatters[idx].set_offsets(np.array([[left_x, left_y]]))
                right_scatters[idx].set_offsets(np.array([[right_x, right_y]]))
                base_size = 100  # adjust as needed
                left_scatters[idx].set_sizes(np.array([base_size * odor_left]))
                right_scatters[idx].set_sizes(np.array([base_size * odor_right]))
                left_scatters[idx].set_array(np.array([odor_left]))
                right_scatters[idx].set_array(np.array([odor_right]))
            time_text.set_text(f"Frame: {frame}")
            return line_collections + left_scatters + right_scatters + [time_text] + heading_arrows
        
        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                      interval=interval, blit=False)
        if save_path:
            ani.save(save_path, writer='imagemagick')
        else:
            plt.show()
        plt.close(fig)
