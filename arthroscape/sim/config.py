# arthroscape/sim/config.py
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class SimulationConfig:
    # Simulation parameters
    T: float = 60 * 60           # Total simulation time in seconds
    fps: float = 60              # Frames per second

    # Motion parameters
    walking_speed: float = 15    # mm/s when walking
    rotation_diffusion: float = np.deg2rad(0.22)  # radians per frame

    # Behavioral algorithm parameters (per second)
    turn_rate: float = 1.0       # Hz, base turning rate
    asymmetry_factor: float = 20 # Increases turn rate when odor asymmetry is high
    turn_magnitude_range: Tuple[float, float] = (np.deg2rad(8), np.deg2rad(30))  # radians

    # State transition rates (per second)
    rate_stop_to_walk: float = 0.5    # Hz, from stop to walking
    rate_walk_to_stop: float = 0.05   # Hz, from walking to stop

    # Odor sensing parameters
    antennal_distance: float = 1.0    # legacy parameter (in mm)
    # New: Antenna offsets in the fly's body frame (dx, dy) in mm.
    antenna_left_offset: Tuple[float, float] = (0.5, 0.5)   # shifted forward & left
    antenna_right_offset: Tuple[float, float] = (0.5, -0.5) # shifted forward & right

    # Odor deposition kernel parameters (for odor release strategies that deposit a spread)
    deposit_sigma: float = 5.0         # Standard deviation (in mm) for the Gaussian deposit
    deposit_kernel_size: int = 20       # Kernel size (number of grid cells, assumed odd)

    # Grid arena parameters
    grid_x_min: float = -80.0
    grid_x_max: float = 80.0
    grid_y_min: float = -80.0
    grid_y_max: float = 80.0
    grid_resolution: float = 0.1       # mm per grid cell

    # Odor history recording parameters (if you wish to record snapshots for animation)
    record_odor_history: bool = False  # Set to True to record odor grid history
    odor_history_interval: int = 100   # Record every N frames

    # Derived parameters (computed automatically)
    walking_distance: float = field(init=False)          # mm per frame
    turn_rate_per_frame: float = field(init=False)         # probability per frame
    asymmetry_factor_per_frame: float = field(init=False)  # probability per frame
    rate_stop_to_walk_per_frame: float = field(init=False) # probability per frame
    rate_walk_to_stop_per_frame: float = field(init=False) # probability per frame
    total_frames: int = field(init=False)

    def __post_init__(self):
        self.walking_distance = self.walking_speed / self.fps
        self.turn_rate_per_frame = self.turn_rate / self.fps
        self.asymmetry_factor_per_frame = self.asymmetry_factor / self.fps
        self.rate_stop_to_walk_per_frame = self.rate_stop_to_walk / self.fps
        self.rate_walk_to_stop_per_frame = self.rate_walk_to_stop / self.fps
        self.total_frames = int(self.T * self.fps)
