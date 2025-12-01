# arthroscape/sim/simulator.py
"""
Core simulation engine for ArthroScape.

This module defines the `MultiAnimalSimulator` class, which orchestrates the simulation loop.
It manages the state of multiple animals, their interactions with the arena (odor deposition,
sensing, movement), and the evolution of the odor field over time.
"""

import math
import logging
import numpy as np
from typing import Any, Dict, List
from .config import SimulationConfig
from .arena import Arena
from .odor_release import OdorReleaseStrategy
from .behavior import BehaviorAlgorithm

logger = logging.getLogger(__name__)


class MultiAnimalSimulator:
    """
    Simulator for multiple animals in an odor arena.

    This class manages the main simulation loop, updating animal states, positions,
    and the odor field at each time step. It supports both a standard sequential
    simulation method and a vectorized version for better performance with many animals.
    """

    def __init__(
        self,
        config: SimulationConfig,
        behavior: BehaviorAlgorithm,
        arena: Arena,
        odor_release_strategy: OdorReleaseStrategy,
        seed: int = None,
    ):
        """
        Initialize the simulator.

        Args:
            config (SimulationConfig): Simulation configuration parameters.
            behavior (BehaviorAlgorithm): The behavior algorithm governing animal decisions.
            arena (Arena): The arena environment (shared by all animals).
            odor_release_strategy (OdorReleaseStrategy): Strategy for odor deposition.
            seed (int, optional): Random seed for the simulation's random number generator.
        """
        self.config = config
        self.behavior = behavior
        self.arena = arena  # shared arena for all animals
        self.odor_release_strategy = odor_release_strategy
        self.rng = np.random.default_rng(seed)
        self.num_animals = config.number_of_animals
        # For each agent, create a new instance of the odor perception:
        self.odor_perceptions = [
            config.odor_perception_factory() for _ in range(self.num_animals)
        ]

    def simulate(self) -> Dict[str, Any]:
        """
        Run the simulation step-by-step (unvectorized).

        This method iterates through each time step and each animal sequentially.
        It performs the following operations in order:
        1. Update animal state (e.g., deciding to turn or walk).
        2. Release odor into the arena based on the current state.
        3. Calculate antenna positions.
        4. Sample odor concentration at antenna positions.
        5. Process odor perception (filtering, adaptation).
        6. Update heading based on perceived odor.
        7. Update position (move) if the path is clear.
        8. Update the global odor field (diffusion and decay).

        Returns:
            Dict[str, Any]: A dictionary containing simulation results, including:
                - "trajectories": A list of dictionaries, one per animal, containing time-series data
                  for position (x, y), heading, state, and odor readings.
                - "final_odor_grid": The state of the odor grid at the end of the simulation.
        """
        cfg = self.config
        N = cfg.total_frames
        num = self.num_animals

        # Initialize per-animal arrays.
        states = [[0] * N for _ in range(num)]
        headings = [[0.0] * N for _ in range(num)]
        xs = [[0.0] * N for _ in range(num)]
        ys = [[0.0] * N for _ in range(num)]
        odor_left_arr = [[0.0] * N for _ in range(num)]
        odor_right_arr = [[0.0] * N for _ in range(num)]
        perc_odor_left_arr = [[0.0] * N for _ in range(num)]
        perc_odor_right_arr = [[0.0] * N for _ in range(num)]

        # Reset each agent's perception at start
        for a in range(num):
            self.odor_perceptions[a].reset()

        # Initialize each animal with a random heading and a random initial position.
        for a in range(num):
            headings[a][0] = cfg.initial_heading_sampler()
            xs[a][0], ys[a][0] = cfg.initial_position_sampler()

        # Main simulation loop.
        progress_interval = max(1, N // 100)  # report every 1%
        for i in range(1, N):
            for a in range(num):
                # 1) Update state
                states[a][i] = self.behavior.update_state(
                    states[a][i - 1], cfg, self.rng
                )

                # 2) Odor release
                deposits = self.odor_release_strategy.release_odor(
                    states[a][i - 1],
                    (xs[a][i - 1], ys[a][i - 1]),
                    headings[a][i - 1],
                    cfg,
                    self.rng,
                )
                for deposit in deposits:
                    dx, dy = deposit.offset
                    heading_prev = headings[a][i - 1]
                    cos_h = math.cos(heading_prev)
                    sin_h = math.sin(heading_prev)
                    global_dx = dx * cos_h - dy * sin_h
                    global_dy = dx * sin_h + dy * cos_h
                    deposit_x = xs[a][i - 1] + global_dx
                    deposit_y = ys[a][i - 1] + global_dy
                    kernel = deposit.generate_kernel(cfg)
                    self.arena.deposit_odor_kernel(deposit_x, deposit_y, kernel)

                # 3) Compute left/right antenna positions
                cos_h = math.cos(headings[a][i - 1])
                sin_h = math.sin(headings[a][i - 1])
                left_dx = (
                    cfg.antenna_left_offset[0] * cos_h
                    - cfg.antenna_left_offset[1] * sin_h
                )
                left_dy = (
                    cfg.antenna_left_offset[0] * sin_h
                    + cfg.antenna_left_offset[1] * cos_h
                )
                right_dx = (
                    cfg.antenna_right_offset[0] * cos_h
                    - cfg.antenna_right_offset[1] * sin_h
                )
                right_dy = (
                    cfg.antenna_right_offset[0] * sin_h
                    + cfg.antenna_right_offset[1] * cos_h
                )
                left_x = xs[a][i - 1] + left_dx
                left_y = ys[a][i - 1] + left_dy
                right_x = xs[a][i - 1] + right_dx
                right_y = ys[a][i - 1] + right_dy

                # 4) Sample odor
                odor_left = self.arena.get_odor(left_x, left_y)
                odor_right = self.arena.get_odor(right_x, right_y)
                odor_left_arr[a][i] = odor_left
                odor_right_arr[a][i] = odor_right

                # 5) Perceive odor
                dt = 1.0 / cfg.fps
                perc_left, perc_right = self.odor_perceptions[a].perceive_odor(
                    odor_left, odor_right, dt
                )
                perc_odor_left_arr[a][i] = perc_left
                perc_odor_right_arr[a][i] = perc_right

                # 6) Update heading
                headings[a][i] = self.behavior.update_heading(
                    headings[a][i - 1], perc_left, perc_right, False, cfg, self.rng
                )

                # 7) Update position if walking
                if states[a][i] == 0:
                    xs[a][i] = xs[a][i - 1]
                    ys[a][i] = ys[a][i - 1]
                else:
                    current_speed = cfg.walking_speed_sampler()
                    walking_distance = current_speed / cfg.fps
                    new_x = xs[a][i - 1] + math.cos(headings[a][i]) * walking_distance
                    new_y = ys[a][i - 1] + math.sin(headings[a][i]) * walking_distance
                    if self.arena.is_free(new_x, new_y):
                        xs[a][i] = new_x
                        ys[a][i] = new_y
                    else:
                        xs[a][i] = xs[a][i - 1]
                        ys[a][i] = ys[a][i - 1]

            # 8) Diffusion / decay
            if cfg.diffusion_coefficient > 0 or cfg.odor_decay_rate > 0:
                dt = 1.0 / cfg.fps
                self.arena.update_odor_field(dt=dt)

            # 9) Progress logging
            if i % progress_interval == 0:
                logger.info(f"Replicate progress: frame {i}/{N} ({i / N:.0%} done)")

        result = {
            "trajectories": [
                {
                    "x": xs[a],
                    "y": ys[a],
                    "heading": headings[a],
                    "state": states[a],
                    "odor_left": odor_left_arr[a],
                    "odor_right": odor_right_arr[a],
                    "perc_odor_left": perc_odor_left_arr[a],
                    "perc_odor_right": perc_odor_right_arr[a],
                }
                for a in range(num)
            ],
            "final_odor_grid": self.arena.odor_grid,
        }
        return result

    def simulate_vectorized(self) -> Dict[str, Any]:
        """
        Run a vectorized version of the simulation.

        This method optimizes the simulation loop by using NumPy vectorization for:
        - Odor deposition (batching deposits).
        - Sensor position calculation (vectorized trigonometry).
        - Odor sampling (vectorized grid access).
        - Position updates (vectorized movement and collision checking).

        Behavioral state updates and heading updates remain per-animal as they may involve
        complex, state-dependent logic that is harder to vectorize efficiently.

        Returns:
            Dict[str, Any]: A dictionary containing simulation results, structured identically
            to `simulate()`.
        """
        cfg = self.config
        N = cfg.total_frames
        num = self.num_animals

        # Pre-allocate arrays for simulation outputs
        states = np.zeros((num, N), dtype=int)
        headings = np.zeros((num, N))
        xs = np.zeros((num, N))
        ys = np.zeros((num, N))
        odor_left_arr = np.zeros((num, N))
        odor_right_arr = np.zeros((num, N))
        perc_odor_left_arr = np.zeros((num, N))
        perc_odor_right_arr = np.zeros((num, N))

        # Initialize positions and headings (still looping over animals for initial values)
        for a in range(num):
            headings[a, 0] = cfg.initial_heading_sampler()
            xs[a, 0], ys[a, 0] = cfg.initial_position_sampler()

        # Reset odor perceptions for each agent
        for a in range(num):
            self.odor_perceptions[a].reset()

        # Main simulation loop (over frames)
        for i in range(1, N):
            # --- State update (scalar per animal) ---
            for a in range(num):
                states[a, i] = self.behavior.update_state(
                    states[a, i - 1], cfg, self.rng
                )

            # --- Odor deposition (vectorized if possible) ---
            deposit_xs = []
            deposit_ys = []
            kernels = []
            for a in range(num):
                deposits = self.odor_release_strategy.release_odor(
                    states[a, i - 1],
                    (xs[a, i - 1], ys[a, i - 1]),
                    headings[a, i - 1],
                    cfg,
                    self.rng,
                )
                for deposit in deposits:
                    dx, dy = deposit.offset
                    # Compute global deposit offset for this animal (scalar)
                    global_dx = dx * math.cos(headings[a, i - 1]) - dy * math.sin(
                        headings[a, i - 1]
                    )
                    global_dy = dx * math.sin(headings[a, i - 1]) + dy * math.cos(
                        headings[a, i - 1]
                    )
                    deposit_xs.append(xs[a, i - 1] + global_dx)
                    deposit_ys.append(ys[a, i - 1] + global_dy)
                    kernels.append(deposit.generate_kernel(cfg))
            # If any deposits occurred, try a vectorized deposit if the arena supports it.
            if deposit_xs:
                deposit_xs_arr = np.array(deposit_xs)
                deposit_ys_arr = np.array(deposit_ys)
                # Check if all kernels have the same shape.
                if all(k.shape == kernels[0].shape for k in kernels) and hasattr(
                    self.arena, "deposit_odor_kernels_vectorized"
                ):
                    self.arena.deposit_odor_kernels_vectorized(
                        deposit_xs_arr, deposit_ys_arr, kernels[0]
                    )
                else:
                    # Fall back to scalar deposits.
                    for dx, dy, kernel in zip(deposit_xs, deposit_ys, kernels):
                        self.arena.deposit_odor_kernel(dx, dy, kernel)

            # --- Compute sensor positions vectorized ---
            # Calculate rotated offsets using vectorized trigonometry.
            current_headings = headings[:, i - 1]  # shape (num,)
            cos_vals = np.cos(current_headings)
            sin_vals = np.sin(current_headings)
            left_offset = np.array(cfg.antenna_left_offset)  # shape (2,)
            right_offset = np.array(cfg.antenna_right_offset)  # shape (2,)
            left_dx = left_offset[0] * cos_vals - left_offset[1] * sin_vals
            left_dy = left_offset[0] * sin_vals + left_offset[1] * cos_vals
            right_dx = right_offset[0] * cos_vals - right_offset[1] * sin_vals
            right_dy = right_offset[0] * sin_vals + right_offset[1] * cos_vals
            left_sensor_x = xs[:, i - 1] + left_dx
            left_sensor_y = ys[:, i - 1] + left_dy
            right_sensor_x = xs[:, i - 1] + right_dx
            right_sensor_y = ys[:, i - 1] + right_dy

            # --- Get odor sensor readings vectorized ---
            # We assume your arena provides a vectorized get_odor_vectorized method.
            if hasattr(self.arena, "get_odor_vectorized"):
                odor_left_vals = self.arena.get_odor_vectorized(
                    left_sensor_x, left_sensor_y
                )
                odor_right_vals = self.arena.get_odor_vectorized(
                    right_sensor_x, right_sensor_y
                )
            else:
                odor_left_vals = np.array(
                    [
                        self.arena.get_odor(x, y)
                        for x, y in zip(left_sensor_x, left_sensor_y)
                    ]
                )
                odor_right_vals = np.array(
                    [
                        self.arena.get_odor(x, y)
                        for x, y in zip(right_sensor_x, right_sensor_y)
                    ]
                )
            odor_left_arr[:, i] = odor_left_vals
            odor_right_arr[:, i] = odor_right_vals

            # --- Update perceived odor (still per-animal) ---
            dt = 1.0 / cfg.fps
            for a in range(num):
                perc_left, perc_right = self.odor_perceptions[a].perceive_odor(
                    odor_left_arr[a, i], odor_right_arr[a, i], dt
                )
                perc_odor_left_arr[a, i] = perc_left
                perc_odor_right_arr[a, i] = perc_right

            # --- Update headings (per-animal) ---
            for a in range(num):
                headings[a, i] = self.behavior.update_heading(
                    headings[a, i - 1],
                    perc_odor_left_arr[a, i],
                    perc_odor_right_arr[a, i],
                    False,
                    cfg,
                    self.rng,
                )

            # --- Compute new positions vectorized ---
            # Compute walking distances for each animal.
            current_speeds = np.array([cfg.walking_speed_sampler() for _ in range(num)])
            walking_distance = current_speeds / cfg.fps
            proposed_x = xs[:, i - 1] + np.cos(headings[:, i]) * walking_distance
            proposed_y = ys[:, i - 1] + np.sin(headings[:, i]) * walking_distance

            # Use vectorized free-space check if available.
            if hasattr(self.arena, "is_free_vectorized"):
                free = self.arena.is_free_vectorized(proposed_x, proposed_y)
            else:
                free = np.array(
                    [self.arena.is_free(x, y) for x, y in zip(proposed_x, proposed_y)]
                )
            xs[:, i] = np.where(free, proposed_x, xs[:, i - 1])
            ys[:, i] = np.where(free, proposed_y, ys[:, i - 1])

            # --- Update the odor field ---
            if cfg.diffusion_coefficient > 0 or cfg.odor_decay_rate > 0:
                self.arena.update_odor_field(dt=dt)

            if i % max(1, N // 100) == 0:
                logger.info(
                    f"Vectorized simulation progress: frame {i}/{N} ({i / N:.0%} done)"
                )

        # Pack the result into a dictionary.
        result = {
            "trajectories": [
                {
                    "x": xs[a, :].tolist(),
                    "y": ys[a, :].tolist(),
                    "heading": headings[a, :].tolist(),
                    "state": states[a, :].tolist(),
                    "odor_left": odor_left_arr[a, :].tolist(),
                    "odor_right": odor_right_arr[a, :].tolist(),
                    "perceived_odor_left": perc_odor_left_arr[a, :].tolist(),
                    "perceived_odor_right": perc_odor_right_arr[a, :].tolist(),
                }
                for a in range(num)
            ],
            "final_odor_grid": self.arena.odor_grid,
        }
        return result
