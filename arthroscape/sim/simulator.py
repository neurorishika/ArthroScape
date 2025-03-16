# arthroscape/sim/simulator.py
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
    def __init__(self, config: SimulationConfig, behavior: BehaviorAlgorithm,
                 arena: Arena, odor_release_strategy: OdorReleaseStrategy, seed: int = None):
        self.config = config
        self.behavior = behavior
        self.arena = arena  # shared arena for all animals
        self.odor_release_strategy = odor_release_strategy
        self.rng = np.random.default_rng(seed)
        self.num_animals = config.number_of_animals
        # For each agent, create a new instance of the odor perception:
        self.odor_perceptions = [config.odor_perception_factory() for _ in range(self.num_animals)]

    def simulate(self) -> Dict[str, Any]:
        """
        Original step-by-step simulation method (unvectorized).
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
                states[a][i] = self.behavior.update_state(states[a][i-1], cfg, self.rng)

                # 2) Odor release
                deposits = self.odor_release_strategy.release_odor(
                    states[a][i-1], (xs[a][i-1], ys[a][i-1]), headings[a][i-1], cfg, self.rng
                )
                for deposit in deposits:
                    dx, dy = deposit.offset
                    heading_prev = headings[a][i-1]
                    cos_h = math.cos(heading_prev)
                    sin_h = math.sin(heading_prev)
                    global_dx = dx * cos_h - dy * sin_h
                    global_dy = dx * sin_h + dy * cos_h
                    deposit_x = xs[a][i-1] + global_dx
                    deposit_y = ys[a][i-1] + global_dy
                    kernel = deposit.generate_kernel(cfg)
                    self.arena.deposit_odor_kernel(deposit_x, deposit_y, kernel)

                # 3) Compute left/right antenna positions
                cos_h = math.cos(headings[a][i-1])
                sin_h = math.sin(headings[a][i-1])
                left_dx = cfg.antenna_left_offset[0] * cos_h - cfg.antenna_left_offset[1] * sin_h
                left_dy = cfg.antenna_left_offset[0] * sin_h + cfg.antenna_left_offset[1] * cos_h
                right_dx = cfg.antenna_right_offset[0] * cos_h - cfg.antenna_right_offset[1] * sin_h
                right_dy = cfg.antenna_right_offset[0] * sin_h + cfg.antenna_right_offset[1] * cos_h
                left_x = xs[a][i-1] + left_dx
                left_y = ys[a][i-1] + left_dy
                right_x = xs[a][i-1] + right_dx
                right_y = ys[a][i-1] + right_dy

                # 4) Sample odor
                odor_left = self.arena.get_odor(left_x, left_y)
                odor_right = self.arena.get_odor(right_x, right_y)
                odor_left_arr[a][i] = odor_left
                odor_right_arr[a][i] = odor_right

                # 5) Perceive odor
                dt = 1.0 / cfg.fps
                perc_left, perc_right = self.odor_perceptions[a].perceive_odor(odor_left, odor_right, dt)
                perc_odor_left_arr[a][i] = perc_left
                perc_odor_right_arr[a][i] = perc_right

                # 6) Update heading
                headings[a][i] = self.behavior.update_heading(
                    headings[a][i-1], perc_left, perc_right, False, cfg, self.rng
                )

                # 7) Update position if walking
                if states[a][i] == 0:
                    xs[a][i] = xs[a][i-1]
                    ys[a][i] = ys[a][i-1]
                else:
                    current_speed = cfg.walking_speed_sampler()
                    walking_distance = current_speed / cfg.fps
                    new_x = xs[a][i-1] + math.cos(headings[a][i]) * walking_distance
                    new_y = ys[a][i-1] + math.sin(headings[a][i]) * walking_distance
                    if self.arena.is_free(new_x, new_y):
                        xs[a][i] = new_x
                        ys[a][i] = new_y
                    else:
                        xs[a][i] = xs[a][i-1]
                        ys[a][i] = ys[a][i-1]

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
                    "perc_odor_right": perc_odor_right_arr[a]
                } for a in range(num)
            ],
            "final_odor_grid": self.arena.odor_grid
        }
        return result

    def simulate_vectorized(self) -> Dict[str, Any]:
        """
        A more vectorized version of the simulation loop.
        Many of the coordinate transformations and odor sampling are done in batch.
        """
        cfg = self.config
        N = cfg.total_frames
        num = self.num_animals

        # Preallocate arrays using NumPy
        states = np.zeros((num, N), dtype=int)
        headings = np.zeros((num, N), dtype=float)
        xs = np.zeros((num, N), dtype=float)
        ys = np.zeros((num, N), dtype=float)
        odor_left_arr = np.zeros((num, N), dtype=float)
        odor_right_arr = np.zeros((num, N), dtype=float)
        perc_left_arr = np.zeros((num, N), dtype=float)
        perc_right_arr = np.zeros((num, N), dtype=float)

        # Reset each agent's perception
        for a in range(num):
            self.odor_perceptions[a].reset()

        # Initialize each agent
        for a in range(num):
            headings[a, 0] = cfg.initial_heading_sampler()
            xs[a, 0], ys[a, 0] = cfg.initial_position_sampler()

        progress_interval = max(1, N // 100)

        for i in range(1, N):
            # Update states (vectorizable over agents)
            for a in range(num):
                states[a, i] = self.behavior.update_state(states[a, i-1], cfg, self.rng)

            # Gather previous positions and headings (vectorized)
            x_prev = xs[:, i-1]
            y_prev = ys[:, i-1]
            heading_prev = headings[:, i-1]

            # ----- Odor Release (still per-agent because deposits vary) -----
            for a in range(num):
                deposits = self.odor_release_strategy.release_odor(
                    states[a, i-1], (x_prev[a], y_prev[a]), heading_prev[a], cfg, self.rng
                )
                for deposit in deposits:
                    dx, dy = deposit.offset
                    cos_h = math.cos(heading_prev[a])
                    sin_h = math.sin(heading_prev[a])
                    global_dx = dx * cos_h - dy * sin_h
                    global_dy = dx * sin_h + dy * cos_h
                    deposit_x = x_prev[a] + global_dx
                    deposit_y = y_prev[a] + global_dy
                    kernel = deposit.generate_kernel(cfg)
                    self.arena.deposit_odor_kernel(deposit_x, deposit_y, kernel)

            # ----- Vectorized computation for antenna positions -----
            cos_h = np.cos(heading_prev)
            sin_h = np.sin(heading_prev)
            left_dx = cfg.antenna_left_offset[0] * cos_h - cfg.antenna_left_offset[1] * sin_h
            left_dy = cfg.antenna_left_offset[0] * sin_h + cfg.antenna_left_offset[1] * cos_h
            right_dx = cfg.antenna_right_offset[0] * cos_h - cfg.antenna_right_offset[1] * sin_h
            right_dy = cfg.antenna_right_offset[0] * sin_h + cfg.antenna_right_offset[1] * cos_h
            left_x = x_prev + left_dx
            left_y = y_prev + left_dy
            right_x = x_prev + right_dx
            right_y = y_prev + right_dy

            # ----- Vectorized odor sampling using new arena method -----
            odor_left_arr[:, i] = self.arena.get_odor_vectorized(left_x, left_y)
            odor_right_arr[:, i] = self.arena.get_odor_vectorized(right_x, right_y)

            # ----- Perception (per-agent loop) -----
            dt = 1.0 / cfg.fps
            for a in range(num):
                pl, pr = self.odor_perceptions[a].perceive_odor(odor_left_arr[a, i], odor_right_arr[a, i], dt)
                perc_left_arr[a, i] = pl
                perc_right_arr[a, i] = pr

            # ----- Heading update (per-agent loop) -----
            for a in range(num):
                headings[a, i] = self.behavior.update_heading(
                    heading_prev[a], perc_left_arr[a, i], perc_right_arr[a, i], False, cfg, self.rng
                )

            # ----- Vectorized position update using vectorized is_free -----
            # Compute potential new positions
            speed = np.array([cfg.walking_speed_sampler() for _ in range(num)])
            dist = speed / cfg.fps
            new_x = xs[:, i-1] + np.cos(headings[:, i]) * dist
            new_y = ys[:, i-1] + np.sin(headings[:, i]) * dist

            # Use the vectorized is_free method to determine which agents can move
            free = self.arena.is_free_vectorized(new_x, new_y)
            xs[:, i] = np.where(free, new_x, xs[:, i-1])
            ys[:, i] = np.where(free, new_y, ys[:, i-1])

            # ----- Diffusion / Decay (vectorized in arena) -----
            if cfg.diffusion_coefficient > 0 or cfg.odor_decay_rate > 0:
                self.arena.update_odor_field(dt=dt)

            if i % progress_interval == 0:
                logger.info(f"[Vectorized] Replicate progress: frame {i}/{N} ({i / N:.0%} done)")

        # Convert results to lists if needed
        def to_list(arr):
            return arr.tolist()

        result = {
            "trajectories": [
                {
                    "x": to_list(xs[a]),
                    "y": to_list(ys[a]),
                    "heading": to_list(headings[a]),
                    "state": to_list(states[a]),
                    "odor_left": to_list(odor_left_arr[a]),
                    "odor_right": to_list(odor_right_arr[a]),
                    "perc_odor_left": to_list(perc_left_arr[a]),
                    "perc_odor_right": to_list(perc_right_arr[a])
                }
                for a in range(num)
            ],
            "final_odor_grid": self.arena.odor_grid
        }
        return result