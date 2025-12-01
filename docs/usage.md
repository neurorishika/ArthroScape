# Usage

This section provides a quick guide on how to run simulations using ArthroScape.

## Running a Basic Simulation

To run a simulation, you typically need to:

1. Define a `SimulationConfig`.
2. Initialize the `Arena`, `OdorReleaseStrategy`, and `BehaviorAlgorithm`.
3. Create a `MultiAnimalSimulator`.
4. Run the simulation.

### Example Script

Here is a minimal example:

```python
from arthroscape.sim.config import SimulationConfig
from arthroscape.sim.arena import Arena
from arthroscape.sim.odor_release import OdorReleaseStrategy
from arthroscape.sim.behavior import BehaviorAlgorithm
from arthroscape.sim.simulator import MultiAnimalSimulator

# 1. Configuration
config = SimulationConfig(
    T=10.0,  # 10 seconds
    number_of_animals=1
)

# 2. Initialize Components (Placeholders - replace with actual implementations)
# Note: You would typically use specific subclasses here.
arena = Arena(config)
odor_strategy = OdorReleaseStrategy(config)
behavior = BehaviorAlgorithm(config)

# 3. Create Simulator
simulator = MultiAnimalSimulator(
    config=config,
    behavior=behavior,
    arena=arena,
    odor_release_strategy=odor_strategy
)

# 4. Run
results = simulator.simulate()
print("Simulation complete.")
```

## Running via Command Line

If the package provides command-line scripts (e.g., in `scripts/`), you can run them using `poetry run`.

```bash
poetry run python scripts/run_simulation.py
```

## Visualizing Results

(Add details about visualization tools here if available)
