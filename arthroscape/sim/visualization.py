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
import cv2  # OpenCV

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
    
    Parameters:
        x, y: 1D arrays of coordinates.
        x_min, x_max, y_min, y_max: Domain boundaries.
        threshold: Tuple (tx, ty) thresholds; if None, defaults to half the domain width/height.
        
    Returns:
        List of (start_idx, end_idx) indices for each continuous segment.
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
        
        for idx, traj in enumerate(result["trajectories"]):
            x = np.array(traj["x"])
            y = np.array(traj["y"])
            
            if wraparound:
                x_wrapped = wrap_coordinates(x, self.config.grid_x_min, self.config.grid_x_max)
                y_wrapped = wrap_coordinates(y, self.config.grid_y_min, self.config.grid_y_max)
                segments_idx = segment_trajectory_with_indices(x_wrapped, y_wrapped,
                                                               self.config.grid_x_min, self.config.grid_x_max,
                                                               self.config.grid_y_min, self.config.grid_y_max)
            else:
                segments_idx = [(0, len(x))]
                x_wrapped = x
                y_wrapped = y

            avg_odor = (np.array(traj["odor_left"]) + np.array(traj["odor_right"])) / 2.0

            for (start, end) in segments_idx:
                if end - start < 2:
                    continue
                segment = np.column_stack((x_wrapped[start:end], y_wrapped[start:end]))
                lc = LineCollection([segment], cmap='viridis', 
                                    norm=plt.Normalize(vmin=avg_odor.min(), vmax=avg_odor.max()))
                lc.set_array(avg_odor[start:end-1])
                lc.set_linewidth(0.5)
                ax.add_collection(lc)
            
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
        """Plot the final odor grid as a heatmap."""
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
        """Plot time series of odor intensity at the antennae for all animals."""
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
        Enhanced animation using Matplotlib: animate trajectories for all animals with heading arrows and antenna markers.
        When wraparound is True, trajectories are pre-segmented to avoid spurious connecting lines.
        """
        result = self.sim_results[sim_index]
        cfg = self.config
        num = len(result["trajectories"])
        colors = plt.cm.rainbow(np.linspace(0, 1, num))
        
        # Pre-compute segmentation indices if wraparound is enabled.
        seg_indices = []
        if wraparound:
            for traj in result["trajectories"]:
                x = np.array(traj["x"])
                y = np.array(traj["y"])
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
        
        # Create a LineCollection for each animal to hold segments.
        line_collections = []
        for idx in range(num):
            lc = LineCollection([], colors=[colors[idx]], linewidths=0.5)
            line_collections.append(lc)
            ax.add_collection(lc)
        
        # Create heading arrows and antenna scatter objects.
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
                    try:
                        heading_arrows[idx].remove()
                    except ValueError:
                        pass
                    heading_arrows[idx] = None
            time_text.set_text("")
            return line_collections + left_scatters + right_scatters + [time_text]
        
        def update(frame):
            for idx, traj in enumerate(result["trajectories"]):
                x = np.array(traj["x"])
                y = np.array(traj["y"])
                if wraparound:
                    x = wrap_coordinates(x, cfg.grid_x_min, cfg.grid_x_max)
                    y = wrap_coordinates(y, cfg.grid_y_min, cfg.grid_y_max)
                segs = seg_indices[idx]
                segments_to_plot = []
                for (start, end) in segs:
                    if frame > start:
                        seg_end = min(end, frame)
                        if seg_end - start > 1:
                            segments_to_plot.append(np.column_stack((x[start:seg_end], y[start:seg_end])))
                line_collections[idx].set_segments(segments_to_plot)
                
                cur_idx = frame - 1
                cur_x = x[cur_idx]
                cur_y = y[cur_idx]
                cur_heading = traj["heading"][cur_idx]
                # Remove previous arrow if it exists.
                if heading_arrows[idx] is not None:
                    try:
                        heading_arrows[idx].remove()
                    except ValueError:
                        pass
                    heading_arrows[idx] = None
                arrow_length = 1.5
                dx = arrow_length * math.cos(cur_heading)
                dy = arrow_length * math.sin(cur_heading)
                heading_arrows[idx] = ax.arrow(cur_x, cur_y, dx, dy, head_width=1, head_length=2,
                                               fc=colors[idx], ec=colors[idx])
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
                base_size = 100
                left_scatters[idx].set_sizes(np.array([base_size * odor_left]))
                right_scatters[idx].set_sizes(np.array([base_size * odor_right]))
                left_scatters[idx].set_array(np.array([odor_left]))
                right_scatters[idx].set_array(np.array([odor_right]))
            time_text.set_text(f"Frame: {frame}")
            return line_collections + left_scatters + right_scatters + [time_text] + heading_arrows
        
        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                      interval=interval, blit=False)
        if save_path:
            ani.save(save_path, writer='imagemagick', fps=1000//interval)
        else:
            plt.show()
        plt.close(fig)

    def animate_enhanced_trajectory_opencv(self, sim_index: int = 0, interval: int = 50, frame_skip: int = 5,
                                        wraparound: bool = False, output_file: str = "animation.mp4",
                                        display: bool = False) -> None:
        """
        OpenCV-based animation: animate trajectories for all animals with ellipse representations
        (colored to match the trajectory) and odor sensor markers.
        
        Parameters:
            sim_index (int): Index of the simulation replicate.
            interval (int): Delay between frames in milliseconds (FPS = 1000/interval).
            frame_skip (int): Process every Nth frame.
            wraparound (bool): If True, wrap and segment trajectories to avoid spurious connecting lines.
            output_file (str): Path to the output video file.
            display (bool): If True, display the animation in an OpenCV window.
        """
        result = self.sim_results[sim_index]
        cfg = self.config
        x_min, x_max = cfg.grid_x_min, cfg.grid_x_max
        y_min, y_max = cfg.grid_y_min, cfg.grid_y_max

        # get grid aspect ratio
        aspect_ratio = (x_max - x_min) / (y_max - y_min)

        # Set a higher resolution (e.g., 1920x1080).
        img_width, img_height = 1920, int(1920 / aspect_ratio)

        def sim_to_pixel(x, y):
            # Map simulation coordinates (x, y) to pixel coordinates (col, row).
            col = int((x - x_min) / (x_max - x_min) * img_width)
            row = img_height - int((y - y_min) / (y_max - y_min) * img_height)
            return col, row

        # Generate distinct colors for each trajectory (BGR for OpenCV)
        cmap = plt.cm.get_cmap('rainbow', len(result["trajectories"]))
        colors = [tuple(int(255 * c) for c in cmap(i)[:3][::-1]) for i in range(len(result["trajectories"]))]

        # Pre-compute segmentation indices if wraparound is enabled.
        seg_indices = []
        if wraparound:
            for traj in result["trajectories"]:
                x_arr = np.array(traj["x"])
                y_arr = np.array(traj["y"])
                x_wrapped = wrap_coordinates(x_arr, x_min, x_max)
                y_wrapped = wrap_coordinates(y_arr, y_min, y_max)
                seg_idx = segment_trajectory_with_indices(x_wrapped, y_wrapped, x_min, x_max, y_min, y_max)
                seg_indices.append(seg_idx)
        else:
            seg_indices = [ [(0, len(traj["x"]))] for traj in result["trajectories"] ]

        # Prepare the VideoWriter.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 1000 // interval  # approximate FPS
        writer = cv2.VideoWriter(output_file, fourcc, fps, (img_width, img_height))

        total_frames = len(result["trajectories"][0]["x"])
        frames = range(0, total_frames, frame_skip)

        # Parameters for drawing the animal's ellipse.
        # Define ellipse axes in pixels (half-lengths).
        ellipse_axes = (20, 10)  # Adjust for a higher resolution display.

        # For time display, compute simulation time from frame index.
        def format_time(frame_index: int) -> str:
            t_seconds = frame_index / cfg.fps
            hours = int(t_seconds // 3600)
            minutes = int((t_seconds % 3600) // 60)
            seconds = t_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

        for frame in frames:
            # Create a blank white image.
            img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

            # Draw trajectories (with thin lines).
            for traj_idx, traj in enumerate(result["trajectories"]):
                x = np.array(traj["x"])
                y = np.array(traj["y"])
                if wraparound:
                    x = wrap_coordinates(x, x_min, x_max)
                    y = wrap_coordinates(y, y_min, y_max)
                segs = seg_indices[traj_idx]
                for (start, end) in segs:
                    if frame > start and (end - start) > 1:
                        seg_end = min(end, frame)
                        pts = np.array([sim_to_pixel(xx, yy) for xx, yy in zip(x[start:seg_end], y[start:seg_end])], dtype=np.int32)
                        if pts.shape[0] >= 2:
                            cv2.polylines(img, [pts], isClosed=False, color=colors[traj_idx],
                                        thickness=1, lineType=cv2.LINE_AA)

            # Draw each animal as an ellipse and sensor markers.
            for traj in result["trajectories"]:
                if frame < 1:
                    continue
                cur_idx = frame - 1
                cur_x = traj["x"][cur_idx]
                cur_y = traj["y"][cur_idx]
                if wraparound:
                    cur_x = wrap_coordinates(np.array([cur_x]), x_min, x_max)[0]
                    cur_y = wrap_coordinates(np.array([cur_y]), y_min, y_max)[0]
                center = sim_to_pixel(cur_x, cur_y)
                cur_heading = traj["heading"][cur_idx]
                # OpenCV's ellipse function expects the angle in degrees (clockwise, so we use negative).
                angle_deg = -math.degrees(cur_heading)
                cv2.ellipse(img, center, ellipse_axes, angle_deg, 0, 360, colors[result["trajectories"].index(traj)], thickness=1, lineType=cv2.LINE_AA)
                
                # Draw sensor markers as circles.
                left_dx = cfg.antenna_left_offset[0] * math.cos(cur_heading) - cfg.antenna_left_offset[1] * math.sin(cur_heading)
                left_dy = cfg.antenna_left_offset[0] * math.sin(cur_heading) + cfg.antenna_left_offset[1] * math.cos(cur_heading)
                right_dx = cfg.antenna_right_offset[0] * math.cos(cur_heading) - cfg.antenna_right_offset[1] * math.sin(cur_heading)
                right_dy = cfg.antenna_right_offset[0] * math.sin(cur_heading) + cfg.antenna_right_offset[1] * math.cos(cur_heading)
                left_x = cur_x + left_dx
                left_y = cur_y + left_dy
                right_x = cur_x + right_dx
                right_y = cur_y + right_dy
                left_center = sim_to_pixel(left_x, left_y)
                right_center = sim_to_pixel(right_x, right_y)
                odor_left = traj["odor_left"][cur_idx]
                odor_right = traj["odor_right"][cur_idx]
                # Scale marker radius by odor intensity.
                radius_left = max(3, int(5 * odor_left))
                radius_right = max(3, int(5 * odor_right))
                cv2.circle(img, left_center, radius_left, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(img, right_center, radius_right, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

            # Draw time information.
            time_str = format_time(frame)
            cv2.putText(img, f"Time: {time_str}", (10, img_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), thickness=2, lineType=cv2.LINE_AA)
            
            if display:
                cv2.imshow('Simulation Animation', img)
                if cv2.waitKey(interval) & 0xFF == ord('q'):
                    break
            
            writer.write(img)

        writer.release()
        if display:
            cv2.destroyAllWindows()
        logger.info(f"OpenCV animation saved to {output_file}")
