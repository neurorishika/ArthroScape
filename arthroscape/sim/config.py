# arthroscape/sim/config.py
import functools
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Sequence, Callable
from .directional_persistence import DirectionalPersistenceStrategy, FixedBlendPersistence
from .odor_perception import AgentOdorPerception, AblatedPerception, NoAdaptationPerception

# Define top-level helper functions that are pickleable.
def _return_walking_speed(walking_speed: float) -> float:
    return walking_speed

def get_walking_speed_sampler(walking_speed: float) -> Callable[[], float]:
    return functools.partial(_return_walking_speed, walking_speed)

def _sample_turn_angle(low: float, high: float) -> float:
    return np.random.uniform(low, high)

def get_turn_angle_sampler(low: float, high: float) -> Callable[[], float]:
    return functools.partial(_sample_turn_angle, low, high)

def _sample_initial_position(grid_x_min: float, grid_x_max: float,
                             grid_y_min: float, grid_y_max: float) -> Tuple[float, float]:
    std_x = (grid_x_max - grid_x_min) / 10
    std_y = (grid_y_max - grid_y_min) / 10
    return (np.random.normal(0, std_x), np.random.normal(0, std_y))

def get_initial_position_sampler(grid_x_min: float, grid_x_max: float,
                                 grid_y_min: float, grid_y_max: float) -> Callable[[], Tuple[float, float]]:
    return functools.partial(_sample_initial_position, grid_x_min, grid_x_max, grid_y_min, grid_y_max)

def _sample_initial_heading() -> float:
    return np.random.uniform(-np.pi, np.pi)

def get_initial_heading_sampler() -> Callable[[], float]:
    return _sample_initial_heading

@dataclass
class SimulationConfig:
    # Simulation parameters
    T: float = 60 * 15         # Total simulation time in seconds
    fps: float = 30              # Frames per second

    # Motion parameters
    walking_speed: float = 15    # mm/s when walking
    rotation_diffusion: float = np.deg2rad(0.22)  # radians per frame
    # Optional sampler for walking speed; if provided, this callable returns a speed at each step.
    walking_speed_sampler: Callable[[], float] = None

    # Behavioral algorithm parameters (per second)
    turn_rate: float = 1.0       # Hz, base turning rate
    asymmetry_factor: float = 10 # Increases turn rate when odor asymmetry is high
    error_rate: float = 0.0         # Hz, Probability of turning in the wrong direction
    odor_driven_turn_scaler: float = 0.0 # Increases the turn angle based on odor asymmetry
    turn_magnitude_range: Tuple[float, float] = (np.deg2rad(8), np.deg2rad(30))  # radians
    # Optional sampler for turn angle (if desired)
    turn_angle_sampler: Callable[[], float] = None

    # State transition rates (per second)
    rate_stop_to_walk: float = 0.5    # Hz, from stop to walking
    rate_walk_to_stop: float = 0.05   # Hz, from walking to stop

    # Odor sensing parameters
    # Antenna offsets in the fly's body frame (dx, dy) in mm.
    antenna_left_offset: Tuple[float, float] = (1.5, 0.5)   # shifted forward & left
    antenna_right_offset: Tuple[float, float] = (1.5, -0.5) # shifted forward & right

    # Odor perception parameters
    odor_perception_factory: Callable[[], AgentOdorPerception] = field(
        default_factory=lambda: NoAdaptationPerception #functools.partial(AblatedPerception, direction="random")
    )

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
    odor_decay_tau: float = 60*2       # Decay time constant in seconds
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
            self.walking_speed_sampler = get_walking_speed_sampler(self.walking_speed)
        if self.turn_angle_sampler is None:
            low, high = self.turn_magnitude_range
            self.turn_angle_sampler = get_turn_angle_sampler(low, high)
        if self.initial_position_sampler is None:
            self.initial_position_sampler = get_initial_position_sampler(
                self.grid_x_min, self.grid_x_max, self.grid_y_min, self.grid_y_max
            )
        if self.initial_heading_sampler is None:
            self.initial_heading_sampler = get_initial_heading_sampler()

        # Compute derived parameters
        self.odor_history_interval_frames = int(self.odor_history_time_interval * self.fps)
            
        self.turn_rate_per_frame = self.turn_rate / self.fps
        self.asymmetry_factor_per_frame = self.asymmetry_factor / self.fps
        self.rate_stop_to_walk_per_frame = self.rate_stop_to_walk / self.fps
        self.rate_walk_to_stop_per_frame = self.rate_walk_to_stop / self.fps
        self.total_frames = int(self.T * self.fps)

        self.error_rate_per_frame = self.error_rate / self.fps

        # Compute kernel size from sigma and deposit_kernel_factor (round up to an odd integer)
        size = int(2 * np.ceil(self.deposit_kernel_factor * self.deposit_sigma / self.grid_resolution)) + 1
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
