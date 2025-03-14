# arthroscape/sim/simulator.py
import math
import logging
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
        self.rng = __import__("numpy").random.default_rng(seed)
        self.num_animals = config.number_of_animals
        # For each agent, create a new instance of the odor perception:
        self.odor_perceptions = [config.odor_perception_factory() for _ in range(self.num_animals)]

    def simulate(self) -> Dict[str, Any]:
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
        for a in range(self.num_animals):
            self.odor_perceptions[a].reset()

        # Initialize each animal with a random heading and a random initial position.
        for a in range(num):
            headings[a][0] = cfg.initial_heading_sampler()
            xs[a][0], ys[a][0] = cfg.initial_position_sampler()

        # Main simulation loop.
        progress_interval = max(1, N // 100)  # report every 1%
        for i in range(1, N):
            for a in range(num):
                states[a][i] = self.behavior.update_state(states[a][i-1], cfg, self.rng)
                deposits = self.odor_release_strategy.release_odor(
                    states[a][i-1], (xs[a][i-1], ys[a][i-1]), headings[a][i-1], cfg, self.rng
                )
                for deposit in deposits:
                    dx, dy = deposit.offset
                    global_dx = dx * math.cos(headings[a][i-1]) - dy * math.sin(headings[a][i-1])
                    global_dy = dx * math.sin(headings[a][i-1]) + dy * math.cos(headings[a][i-1])
                    deposit_x = xs[a][i-1] + global_dx
                    deposit_y = ys[a][i-1] + global_dy
                    kernel = deposit.generate_kernel(cfg)
                    self.arena.deposit_odor_kernel(deposit_x, deposit_y, kernel)

                left_offset = cfg.antenna_left_offset
                right_offset = cfg.antenna_right_offset
                left_global_dx = left_offset[0] * math.cos(headings[a][i-1]) - left_offset[1] * math.sin(headings[a][i-1])
                left_global_dy = left_offset[0] * math.sin(headings[a][i-1]) + left_offset[1] * math.cos(headings[a][i-1])
                right_global_dx = right_offset[0] * math.cos(headings[a][i-1]) - right_offset[1] * math.sin(headings[a][i-1])
                right_global_dy = right_offset[0] * math.sin(headings[a][i-1]) + right_offset[1] * math.cos(headings[a][i-1])
                left_x = xs[a][i-1] + left_global_dx
                left_y = ys[a][i-1] + left_global_dy
                right_x = xs[a][i-1] + right_global_dx
                right_y = ys[a][i-1] + right_global_dy

                odor_left = self.arena.get_odor(left_x, left_y)
                odor_right = self.arena.get_odor(right_x, right_y)
                odor_left_arr[a][i] = odor_left
                odor_right_arr[a][i] = odor_right

                # Update the perceived odor values for this agent.
                dt = 1.0 / cfg.fps  # time step per frame
                perc_odor_left, perc_odor_right = self.odor_perceptions[a].perceive_odor(odor_left, odor_right, dt)
                perc_odor_left_arr[a][i] = perc_odor_left
                perc_odor_right_arr[a][i] = perc_odor_right

                headings[a][i] = self.behavior.update_heading(
                    headings[a][i-1], perc_odor_left, perc_odor_right, False, cfg, self.rng
                )
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
            
            if cfg.diffusion_coefficient > 0 or cfg.odor_decay_rate > 0:
                dt = 1.0 / cfg.fps  # time step per frame
                self.arena.update_odor_field(dt=dt)

            # Log progress every progress_interval frames.
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