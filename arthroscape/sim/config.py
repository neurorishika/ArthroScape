# arthroscape/sim/config.py
"""
Configuration module for the ArthroScape simulation.

This module defines the `SimulationConfig` dataclass, which holds all parameters
controlling the simulation environment, agent behavior, odor dynamics, and more.
It also provides helper functions for sampling initial conditions and movement parameters.
"""

import functools
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Sequence, Callable
from .directional_persistence import (
    DirectionalPersistenceStrategy,
    FixedBlendPersistence,
)
from .odor_perception import (
    AgentOdorPerception,
    AblatedPerception,
    NoAdaptationPerception,
    LowPassPerception,
    LeakAdaptationPerception,
    AblatedLeakAdaptationPerception,
)


# Define top-level helper functions that are pickleable.
def _return_walking_speed(walking_speed: float) -> float:
    """Helper function to return a constant walking speed."""
    return walking_speed


def get_walking_speed_sampler(walking_speed: float) -> Callable[[], float]:
    """
    Returns a callable that returns a constant walking speed.

    Args:
        walking_speed (float): The walking speed to return.

    Returns:
        Callable[[], float]: A function that returns the walking speed.
    """
    return functools.partial(_return_walking_speed, walking_speed)


def _sample_turn_angle(low: float, high: float) -> float:
    """Helper function to sample a turn angle from a uniform distribution."""
    return np.random.uniform(low, high)


def get_turn_angle_sampler(low: float, high: float) -> Callable[[], float]:
    """
    Returns a callable that samples a turn angle from a uniform distribution.

    Args:
        low (float): The lower bound of the distribution (radians).
        high (float): The upper bound of the distribution (radians).

    Returns:
        Callable[[], float]: A function that returns a sampled turn angle.
    """
    return functools.partial(_sample_turn_angle, low, high)


def _sample_initial_position(
    grid_x_min: float, grid_x_max: float, grid_y_min: float, grid_y_max: float
) -> Tuple[float, float]:
    """Helper function to sample an initial position from a normal distribution centered at (0,0)."""
    std_x = (grid_x_max - grid_x_min) / 10
    std_y = (grid_y_max - grid_y_min) / 10
    return (np.random.normal(0, std_x), np.random.normal(0, std_y))


def get_initial_position_sampler(
    grid_x_min: float, grid_x_max: float, grid_y_min: float, grid_y_max: float
) -> Callable[[], Tuple[float, float]]:
    """
    Returns a callable that samples an initial position.

    The position is sampled from a normal distribution centered at (0, 0) with
    standard deviation equal to 1/10th of the grid dimensions.

    Args:
        grid_x_min (float): Minimum x-coordinate of the grid.
        grid_x_max (float): Maximum x-coordinate of the grid.
        grid_y_min (float): Minimum y-coordinate of the grid.
        grid_y_max (float): Maximum y-coordinate of the grid.

    Returns:
        Callable[[], Tuple[float, float]]: A function that returns a tuple (x, y).
    """
    return functools.partial(
        _sample_initial_position, grid_x_min, grid_x_max, grid_y_min, grid_y_max
    )


def _sample_initial_heading() -> float:
    """Helper function to sample an initial heading from a uniform distribution [-pi, pi]."""
    return np.random.uniform(-np.pi, np.pi)


def get_initial_heading_sampler() -> Callable[[], float]:
    """
    Returns a callable that samples an initial heading.

    The heading is sampled uniformly from [-pi, pi].

    Returns:
        Callable[[], float]: A function that returns a sampled heading in radians.
    """
    return _sample_initial_heading


@dataclass
class SimulationConfig:
    """
    Configuration class for the ArthroScape simulation.

    This dataclass contains all the parameters required to set up and run a simulation,
    including time settings, agent motion parameters, behavioral rules, odor perception
    settings, and arena properties.

    Attributes:
        T (float): Total simulation time in seconds. Default is 900 (15 minutes).
        fps (float): Frames per second for the simulation. Default is 60.
        walking_speed (float): Walking speed of the agent in mm/s. Default is 15.
        rotation_diffusion (float): Rotational diffusion (random noise in heading) in radians per frame. Default is ~0.22 degrees.
        walking_speed_sampler (Callable[[], float]): Optional callable that returns a walking speed at each step.
        turn_rate (float): Base turning rate in Hz (turns per second). Default is 1.0.
        asymmetry_factor (float): Factor that increases turn rate based on odor asymmetry. Default is 10.
        error_rate (float): Probability (per second) of turning in the wrong direction (against the gradient). Default is 0.
        odor_driven_turn_scaler (float): Factor that scales the turn angle based on odor asymmetry. Default is 0.0.
        turn_magnitude_range (Tuple[float, float]): Range (min, max) of turn angles in radians. Default is (8 deg, 30 deg).
        turn_angle_sampler (Callable[[], float]): Optional callable that returns a turn angle.
        rate_stop_to_walk (float): Rate (Hz) of transition from stop to walk state. Default is 0.5.
        rate_walk_to_stop (float): Rate (Hz) of transition from walk to stop state. Default is 0.05.
        antenna_left_offset (Tuple[float, float]): Position of the left antenna relative to body center (dx, dy) in mm.
        antenna_right_offset (Tuple[float, float]): Position of the right antenna relative to body center (dx, dy) in mm.
        odor_perception_factory (Callable[[], AgentOdorPerception]): Factory function to create odor perception objects.
        deposit_sigma (float): Standard deviation (mm) for the Gaussian odor deposit. Default is 1.0.
        deposit_kernel_factor (float): Number of sigmas to cover in the deposit kernel. Default is 3.0.
        deposit_kernel_size (int): Size of the deposit kernel (computed automatically).
        odor_deposit_offsets (Sequence[Tuple[float, float]]): List of offsets (dx, dy) for odor deposition relative to body center.
        clamp_odor_mask (bool): If True, wall/odor mask cells are clamped. Default is False.
        diffusion_coefficient (float): Diffusion coefficient for odor in mm^2/s. Default is 0.0.
        odor_decay_tau (float): Time constant for odor decay in seconds. Default is infinity (no decay).
        odor_decay_rate (float): Decay rate per frame (computed automatically).
        grid_x_min (float): Minimum x-coordinate of the arena grid. Default is -80.0.
        grid_x_max (float): Maximum x-coordinate of the arena grid. Default is 80.0.
        grid_y_min (float): Minimum y-coordinate of the arena grid. Default is -80.0.
        grid_y_max (float): Maximum y-coordinate of the arena grid. Default is 80.0.
        grid_resolution (float): Resolution of the grid in mm per cell. Default is 0.1.
        record_odor_history (bool): If True, records the history of the odor grid. Default is False.
        odor_history_time_interval (int): Interval in seconds between odor history snapshots. Default is 1.
        odor_history_interval_frames (int): Interval in frames between odor history snapshots (computed automatically).
        number_of_animals (int): Number of agents in the simulation. Default is 1.
        initial_position_sampler (Callable[[], Tuple[float, float]]): Callable to sample initial positions.
        initial_heading_sampler (Callable[[], float]): Callable to sample initial headings.
        directional_persistence_strategy (DirectionalPersistenceStrategy): Strategy for directional persistence.
        turn_rate_per_frame (float): Turn probability per frame (computed).
        asymmetry_factor_per_frame (float): Asymmetry factor per frame (computed).
        rate_stop_to_walk_per_frame (float): Stop-to-walk probability per frame (computed).
        rate_walk_to_stop_per_frame (float): Walk-to-stop probability per frame (computed).
        total_frames (int): Total number of simulation frames (computed).
    """

    # Simulation parameters
    T: float = 60 * 15  # Total simulation time in seconds
    fps: float = 60  # Frames per second

    # Motion parameters
    walking_speed: float = 15  # mm/s when walking
    rotation_diffusion: float = np.deg2rad(0.22)  # radians per frame
    # Optional sampler for walking speed; if provided, this callable returns a speed at each step.
    walking_speed_sampler: Callable[[], float] = None

    # Behavioral algorithm parameters (per second)
    turn_rate: float = 1.0  # Hz, base turning rate
    asymmetry_factor: float = 10  # Increases turn rate when odor asymmetry is high
    error_rate: float = 0  # Hz, Probability of turning in the wrong direction
    odor_driven_turn_scaler: float = (
        0.0  # Increases the turn angle based on odor asymmetry
    )
    turn_magnitude_range: Tuple[float, float] = (
        np.deg2rad(8),
        np.deg2rad(30),
    )  # radians
    # Optional sampler for turn angle (if desired)
    turn_angle_sampler: Callable[[], float] = None

    # State transition rates (per second)
    rate_stop_to_walk: float = 0.5  # Hz, from stop to walking
    rate_walk_to_stop: float = 0.05  # Hz, from walking to stop

    # Odor sensing parameters
    # Antenna offsets in the fly's body frame (dx, dy) in mm.
    antenna_left_offset: Tuple[float, float] = (1.5, 0.125)  # shifted forward & left
    antenna_right_offset: Tuple[float, float] = (1.5, -0.125)  # shifted forward & right

    # Odor perception parameters
    odor_perception_factory: Callable[[], AgentOdorPerception] = field(
        default_factory=lambda: NoAdaptationPerception
        # default_factory=lambda: functools.partial(LowPassPerception, tau=0.05)  # 50 ms
        # default_factory=lambda: functools.partial(LeakAdaptationPerception, odor_integration_tau=0.02, adaptation_tau=0.1, adaptation_magnitude=2.0)
        # default_factory=lambda: functools.partial(AblatedLeakAdaptationPerception, odor_integration_tau=0.02, adaptation_tau=0.1, adaptation_magnitude=2.0, direction='left')
        # default_factory=lambda: functools.partial(AblatedPerception, direction='left')
    )

    # Odor deposition kernel parameters
    deposit_sigma: float = 1.0  # Standard deviation (in mm) for the Gaussian deposit
    deposit_kernel_factor: float = 3.0  # How many sigma to cover on each side
    deposit_kernel_size: int = field(init=False)  # Computed automatically

    # Odor deposition offsets (relative to the fly's centroid)
    odor_deposit_offsets: Sequence[Tuple[float, float]] = (
        (-1.5, 0),
    )  # e.g. deposit behind the fly

    # Odor mask clamping parameters.
    clamp_odor_mask: bool = False  # If True, wall/odor mask cells are clamped.

    # Dynamic odor field parameters
    diffusion_coefficient: float = 0.0  # Diffusion coefficient (in mm^2/s)
    odor_decay_tau: float = np.inf  # Decay time constant in seconds
    odor_decay_rate: float = field(init=False)  # per frame

    # Grid arena parameters
    grid_x_min: float = -80.0
    grid_x_max: float = 80.0
    grid_y_min: float = -80.0
    grid_y_max: float = 80.0
    grid_resolution: float = 0.1  # mm per grid cell

    # Odor history recording parameters
    record_odor_history: bool = False  # Set True to record odor grid history
    odor_history_time_interval: int = 1  # seconds between history snapshots
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
    turn_rate_per_frame: float = field(init=False)  # probability per frame
    asymmetry_factor_per_frame: float = field(init=False)  # probability per frame
    rate_stop_to_walk_per_frame: float = field(init=False)  # probability per frame
    rate_walk_to_stop_per_frame: float = field(init=False)  # probability per frame
    total_frames: int = field(init=False)

    def __post_init__(self):
        """
        Post-initialization processing.

        This method initializes samplers if they are not provided and computes
        derived parameters such as per-frame rates and kernel sizes.
        """
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
        self.odor_history_interval_frames = int(
            self.odor_history_time_interval * self.fps
        )

        self.turn_rate_per_frame = self.turn_rate / self.fps
        self.asymmetry_factor_per_frame = self.asymmetry_factor / self.fps
        self.rate_stop_to_walk_per_frame = self.rate_stop_to_walk / self.fps
        self.rate_walk_to_stop_per_frame = self.rate_walk_to_stop / self.fps
        self.total_frames = int(self.T * self.fps)

        self.error_rate_per_frame = self.error_rate / self.fps

        # Compute kernel size from sigma and deposit_kernel_factor (round up to an odd integer)
        size = (
            int(
                2
                * np.ceil(
                    self.deposit_kernel_factor
                    * self.deposit_sigma
                    / self.grid_resolution
                )
            )
            + 1
        )
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
