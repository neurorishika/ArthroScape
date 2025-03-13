# arthroscape/sim/config.py
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Sequence, Callable
from .directional_persistence import DirectionalPersistenceStrategy

@dataclass
class SimulationConfig:
    # Simulation parameters
    T: float = 60 * 5          # Total simulation time in seconds
    fps: float = 60              # Frames per second

    # Motion parameters
    walking_speed: float = 15    # mm/s when walking
    rotation_diffusion: float = np.deg2rad(0.22)  # radians per frame
    # Optional sampler for walking speed; if provided, this callable returns a speed at each step.
    walking_speed_sampler: Callable[[], float] = None

    # Behavioral algorithm parameters (per second)
    turn_rate: float = 1.0       # Hz, base turning rate
    asymmetry_factor: float = 20 # Increases turn rate when odor asymmetry is high
    turn_magnitude_range: Tuple[float, float] = (np.deg2rad(8), np.deg2rad(30))  # radians
    # Optional sampler for turn angle (if desired)
    turn_angle_sampler: Callable[[], float] = None

    # State transition rates (per second)
    rate_stop_to_walk: float = 0.5    # Hz, from stop to walking
    rate_walk_to_stop: float = 0.05   # Hz, from walking to stop

    # Odor sensing parameters
    antennal_distance: float = 1.0    # legacy parameter (in mm)
    # Antenna offsets in the fly's body frame (dx, dy) in mm.
    antenna_left_offset: Tuple[float, float] = (1.5, 0.5)   # shifted forward & left
    antenna_right_offset: Tuple[float, float] = (1.5, -0.5) # shifted forward & right

    # Odor deposition kernel parameters
    deposit_sigma: float = 1.0         # Standard deviation (in mm) for the Gaussian deposit
    deposit_kernel_factor: float = 3.0 # How many sigma to cover on each side
    deposit_kernel_size: int = field(init=False)  # Computed automatically

    # Odor deposition offsets (relative to the fly's centroid)
    odor_deposit_offsets: Sequence[Tuple[float, float]] = ((-1.5, 0),)  # e.g. deposit behind the fly

    # Odor mask clamping parameters.
    clamp_odor_mask: bool = False      # If True, wall/odor mask cells are clamped.

    # Dynamic odor field parameters
    diffusion_coefficient: float = 0.0  # Diffusion coefficient (in mm^2/s)
    odor_decay_tau: float = np.inf        # Decay time constant in seconds
    odor_decay_rate: float = field(init=False)             # per frame

    # Grid arena parameters
    grid_x_min: float = -80.0
    grid_x_max: float = 80.0
    grid_y_min: float = -80.0
    grid_y_max: float = 80.0
    grid_resolution: float = 0.1       # mm per grid cell

    # Odor history recording parameters
    record_odor_history: bool = False  # Set True to record odor grid history
    odor_history_time_interval: int =  1 # seconds between history snapshots
    odor_history_interval_frames: int = field(init=False)  # Computed automatically

    # Number of animals
    number_of_animals: int = 1

    # Initial position sampler.
    # A callable that returns a tuple (x, y). If None, defaults to a normal distribution. with mean at (0, 0) and std grid_width/5
    initial_position_sampler: Callable[[], Tuple[float, float]] = None

    # Initial heading sampler.
    # A callable that returns a heading angle in radians. If None, defaults to a uniform sampler.
    initial_heading_sampler: Callable[[], float] = None

    # New: Directional persistence strategy (if None, default fixed blend is used).
    directional_persistence_strategy: DirectionalPersistenceStrategy = None
        
    # Derived parameters
    turn_rate_per_frame: float = field(init=False)         # probability per frame
    asymmetry_factor_per_frame: float = field(init=False)  # probability per frame
    rate_stop_to_walk_per_frame: float = field(init=False) # probability per frame
    rate_walk_to_stop_per_frame: float = field(init=False) # probability per frame
    total_frames: int = field(init=False)

    def __post_init__(self):
        if self.walking_speed_sampler is None:
            # Default: fixed walking speed
            self.walking_speed_sampler = lambda: self.walking_speed
        if self.turn_angle_sampler is None:
            # Default: sample uniformly between the given range
            low, high = self.turn_magnitude_range
            self.turn_angle_sampler = lambda: np.random.uniform(low, high)
        if self.initial_position_sampler is None:
            # Default: sample from a normal distribution with std = grid_width/5
            self.initial_position_sampler = lambda: (np.random.normal(0, (self.grid_x_max - self.grid_x_min) / 10),
                                                     np.random.normal(0, (self.grid_y_max - self.grid_y_min) / 10))
        if self.initial_heading_sampler is None:
            # Default: sample uniformly between -pi and pi
            self.initial_heading_sampler = lambda: np.random.uniform(-np.pi, np.pi) # radians

        # Compute derived parameters
        self.odor_history_interval_frames = int(self.odor_history_time_interval * self.fps)
            
        self.turn_rate_per_frame = self.turn_rate / self.fps
        self.asymmetry_factor_per_frame = self.asymmetry_factor / self.fps
        self.rate_stop_to_walk_per_frame = self.rate_stop_to_walk / self.fps
        self.rate_walk_to_stop_per_frame = self.rate_walk_to_stop / self.fps
        self.total_frames = int(self.T * self.fps)

        # Compute kernel size from sigma and deposit_kernel_factor (round up to an odd integer)
        size = int(2 * np.ceil(self.deposit_kernel_factor * self.deposit_sigma / self.grid_resolution)) + 1
        print("size", size)
        self.deposit_kernel_size = size

        # Compute odor decay rate per frame from the time constant
        if self.odor_decay_tau == np.inf:
            self.odor_decay_rate = 0.0
        else:
            self.odor_decay_rate = 1 - np.exp(-1.0 / self.fps / self.odor_decay_tau)

        if self.directional_persistence_strategy is None:
            # Default to a fixed blend with 50% persistence.
            from .directional_persistence import FixedBlendPersistence
            self.directional_persistence_strategy = FixedBlendPersistence(alpha=0.0)
