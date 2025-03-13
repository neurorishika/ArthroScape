# arthroscape/sim/simulator.py
import math
from typing import Any, Dict
from .config import SimulationConfig
from .arena import Arena
from .odor_release import OdorReleaseStrategy
from .behavior import BehaviorAlgorithm

class FruitFlySimulator:
    def __init__(self, config: SimulationConfig, behavior: BehaviorAlgorithm,
                 arena: Arena, odor_release_strategy: OdorReleaseStrategy, seed: int = None):
        self.config = config
        self.behavior = behavior
        self.arena = arena
        self.odor_release_strategy = odor_release_strategy
        self.rng = __import__("numpy").random.default_rng(seed)

    def simulate(self) -> Dict[str, Any]:
        N = self.config.total_frames
        state = [0] * N
        heading = [0.0] * N
        x = [0.0] * N
        y = [0.0] * N
        odor_left_arr = [0.0] * N
        odor_right_arr = [0.0] * N

        heading[0] = self.rng.uniform(0, 2 * math.pi)
        at_wall = False

        for i in range(1, N):
            state[i] = self.behavior.update_state(state[i-1], self.config, self.rng)
            # Get deposit instructions from the odor release strategy.
            deposits = self.odor_release_strategy.release_odor(state[i-1],
                                                               (x[i-1], y[i-1]),
                                                               heading[i-1],
                                                               self.config,
                                                               self.rng)
            for deposit in deposits:
                dx, dy = deposit.offset
                # Rotate the deposit offset by the current heading.
                global_dx = dx * math.cos(heading[i-1]) - dy * math.sin(heading[i-1])
                global_dy = dx * math.sin(heading[i-1]) + dy * math.cos(heading[i-1])
                deposit_x = x[i-1] + global_dx
                deposit_y = y[i-1] + global_dy
                kernel = deposit.generate_kernel(self.config)
                self.arena.deposit_odor_kernel(deposit_x, deposit_y, kernel)

            # Compute antenna positions using configurable offsets.
            left_offset = self.config.antenna_left_offset
            right_offset = self.config.antenna_right_offset
            left_global_dx = left_offset[0] * math.cos(heading[i-1]) - left_offset[1] * math.sin(heading[i-1])
            left_global_dy = left_offset[0] * math.sin(heading[i-1]) + left_offset[1] * math.cos(heading[i-1])
            right_global_dx = right_offset[0] * math.cos(heading[i-1]) - right_offset[1] * math.sin(heading[i-1])
            right_global_dy = right_offset[0] * math.sin(heading[i-1]) + right_offset[1] * math.cos(heading[i-1])
            left_x = x[i-1] + left_global_dx
            left_y = y[i-1] + left_global_dy
            right_x = x[i-1] + right_global_dx
            right_y = y[i-1] + right_global_dy

            odor_left = self.arena.get_odor(left_x, left_y)
            odor_right = self.arena.get_odor(right_x, right_y)
            odor_left_arr[i] = odor_left
            odor_right_arr[i] = odor_right

            heading[i] = self.behavior.update_heading(heading[i-1], odor_left, odor_right,
                                                       at_wall, self.config, self.rng)

            if state[i] == 0:
                x[i] = x[i-1]
                y[i] = y[i-1]
            else:
                new_x = x[i-1] + math.cos(heading[i]) * self.config.walking_distance
                new_y = y[i-1] + math.sin(heading[i]) * self.config.walking_distance
                if self.arena.is_free(new_x, new_y):
                    x[i] = new_x
                    y[i] = new_y
                    at_wall = False
                else:
                    x[i] = x[i-1]
                    y[i] = y[i-1]
                    at_wall = True

        return {
            "x": x,
            "y": y,
            "heading": heading,
            "state": state,
            "odor_left": odor_left_arr,
            "odor_right": odor_right_arr,
            "final_odor_grid": self.arena.odor_grid
        }
