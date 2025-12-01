# Creating Custom Behaviors

ArthroScape is designed to be extensible. You can define your own behavioral algorithms to test different hypotheses about agent movement.

## The `BehaviorAlgorithm` Class

All behaviors must inherit from the abstract base class `BehaviorAlgorithm` defined in `arthroscape.sim.behavior`.

You need to implement two methods:

1. `update_state`: Determines if the agent should stop or walk.
2. `update_heading`: Determines the agent's new heading.

## Example: Random Walker

Let's implement a simple random walker that ignores odor.

```python
import numpy as np
from arthroscape.sim.behavior import BehaviorAlgorithm
from arthroscape.sim.config import SimulationConfig

class RandomWalker(BehaviorAlgorithm):
    def update_state(self, prev_state: int, config: SimulationConfig, rng: np.random.Generator) -> int:
        # Simple state switching based on probabilities
        if prev_state == 0: # Stopped
            if rng.random() < config.rate_stop_to_walk_per_frame:
                return 1 # Start walking
            else:
                return 0 # Stay stopped
        else: # Walking
            if rng.random() < config.rate_walk_to_stop_per_frame:
                return 0 # Stop
            else:
                return 1 # Keep walking

    def update_heading(self, prev_heading: float, odor_left: float, odor_right: float,
                       at_wall: bool, config: SimulationConfig, rng: np.random.Generator) -> float:
        # Ignore odor, just add random noise (diffusion)
        # If at wall, turn around randomly
        
        if at_wall:
            return rng.uniform(-np.pi, np.pi)
            
        # Add rotational diffusion
        noise = rng.normal(0, config.rotation_diffusion)
        
        # Occasionally make a sharp turn
        if rng.random() < config.turn_rate_per_frame:
            turn = config.turn_angle_sampler()
            if rng.random() < 0.5:
                turn = -turn
            return prev_heading + turn + noise
            
        return prev_heading + noise
```

## Using Your Custom Behavior

Once defined, you can pass your new class to the simulation runner.

```python
from arthroscape.sim.runner import run_simulations_vectorized

# ... setup config and arena ...

behavior = RandomWalker()

results = run_simulations_vectorized(
    config, 
    behavior, 
    arena, 
    odor_release, 
    n_replicates=1
)
```

## Advanced: Odor-Driven Behavior

To implement chemotaxis, use the `odor_left` and `odor_right` arguments in `update_heading`.

```python
    def update_heading(self, prev_heading, odor_left, odor_right, at_wall, config, rng):
        # ...
        
        # Calculate contrast
        contrast = (odor_left - odor_right) / (odor_left + odor_right + 1e-6)
        
        # Turn towards higher concentration
        turn_bias = contrast * config.asymmetry_factor
        
        # ... apply turn_bias to heading ...
```
