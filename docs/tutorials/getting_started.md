# Getting Started with ArthroScape

This tutorial will guide you through running your first simulation with ArthroScape. You will learn how to run simulations using both the command-line interface (CLI) and the Python API.

## Prerequisites

Ensure you have installed ArthroScape as described in the [Installation](../installation.md) guide.

## Running a Simulation via CLI

The easiest way to start is using the provided CLI tool.

1. **Open your terminal.**
2. **Run a basic simulation:**

    ```bash
    python -m arthroscape.sim.main --arena circular --animals 5 --visualize
    ```

    This command does the following:
    * Selects a `circular` arena.
    * Places `5` animals in the arena.
    * Enables visualization (`--visualize`), which will generate plots and animations after the simulation finishes.

3. **Check the output:**
    
    By default, results are saved in `data/simulation/<date>/<time>_<description>/`.
    You should see a `results.h5` file and a `plots/` directory containing trajectories and odor grid visualizations.

## Running a Simulation via Python

For more control, you can run simulations directly from a Python script.

1. **Create a new Python file** (e.g., `my_simulation.py`).
2. **Import the necessary modules:**

    ```python
    from arthroscape.sim.config import SimulationConfig
    from arthroscape.sim.arena import create_circular_arena_with_annular_trail
    from arthroscape.sim.behavior import DefaultBehavior
    from arthroscape.sim.odor_release import ConstantOdorRelease
    from arthroscape.sim.runner import run_simulations_vectorized
    from arthroscape.sim.visualization import VisualizationPipeline
    ```

3. **Configure the simulation:**

    ```python
    # Create a configuration object
    config = SimulationConfig(
        T=300,              # 5 minutes
        number_of_animals=10,
        walking_speed=15.0
    )
    ```

4. **Set up the environment:**

    ```python
    # Create an arena
    arena = create_circular_arena_with_annular_trail(
        config, 
        arena_radius=100.0, 
        trail_radius=60.0
    )

    # Define behavior and odor release
    behavior = DefaultBehavior()
    odor_release = ConstantOdorRelease(config, deposit_amount=0.5)
    ```

5. **Run the simulation:**

    ```python
    print("Running simulation...")
    results = run_simulations_vectorized(
        config, 
        behavior, 
        arena, 
        odor_release, 
        n_replicates=1
    )
    print("Simulation complete!")
    ```

6. **Visualize the results:**

    ```python
    pipeline = VisualizationPipeline(results, config, arena)
    pipeline.plot_trajectories_with_odor(sim_index=0, show=True)
    ```

## Next Steps

Now that you can run a basic simulation, try:

* [Customizing the Arena](custom_arena.md) to create different environments.
* [Creating Custom Behaviors](custom_behavior.md) to test new algorithms.
