# Architecture & Internals

This document explains the internal design of ArthroScape. It is intended as a reference for developers and coding agents who need to understand the flow of information and code organization.

## High-Level Overview

ArthroScape is a **vectorized, agent-based simulation framework**. It models the interaction between agents (e.g., flies) and a dynamic environment (arena with odor).

Key design principles:

1. **Separation of Concerns**: Physics (`Arena`), Decision Making (`Behavior`), and State Management (`Simulator`) are decoupled.
2. **Vectorization**: To support efficient simulation of many agents, core loops use NumPy arrays and Numba JIT compilation instead of iterating over agent objects.
3. **Configuration-Driven**: A single `SimulationConfig` object controls almost all aspects of the simulation.

## Code Organization

The core logic resides in `arthroscape/sim/`.

| Module | Role | Key Classes/Functions |
| :--- | :--- | :--- |
| `config.py` | **Configuration**. Defines the schema for simulation parameters. | `SimulationConfig` |
| `simulator.py` | **State Management**. Holds the state of agents and orchestrates the simulation loop. | `MultiAnimalSimulator` |
| `arena.py` | **Environment**. Manages spatial boundaries, obstacles, and the odor grid. Handles coordinate transforms. | `Arena`, `GridArena`, `PeriodicSquareArena` |
| `behavior.py` | **Decision Making**. Determines how agents update their state and heading based on inputs. | `BehaviorAlgorithm`, `DefaultBehavior` |
| `odor_perception.py` | **Sensing**. Models the processing of raw odor signals (filtering, adaptation). | `AgentOdorPerception`, `LowPassPerception` |
| `odor_release.py` | **Stimulus Generation**. Controls how odor is added to the environment (by agents or static sources). | `OdorReleaseStrategy`, `ConstantOdorRelease` |
| `runner.py` | **Execution**. High-level functions to run simulations (serial or parallel). | `run_simulations_vectorized` |
| `visualization.py` | **Analysis**. Plotting and animation tools. | `VisualizationPipeline` |

## The Simulation Loop

The simulation proceeds in discrete time steps. The `MultiAnimalSimulator.step()` method is the heartbeat of the system.

### 1. Initialization

* **Config**: A `SimulationConfig` is created.
* **Arena**: The `Arena` is initialized (grid size, walls).
* **Agents**: Initial positions and headings are sampled. Arrays for `x`, `y`, `heading`, `state` are allocated.

### 2. Update Cycle (Per Frame)

In each frame, the following sequence occurs:

#### A. Odor Dynamics

1. **Release**: The `OdorReleaseStrategy` calculates where odor should be deposited (e.g., at current agent positions).
2. **Deposition**: The `Arena` updates the odor grid with these deposits.
3. **Physics**: The `Arena` applies diffusion and decay to the odor grid.

#### B. Sensory Input

1. **Sensing**: Agents query the `Arena` to get odor concentrations at their left and right antennae positions.
2. **Perception**: The `AgentOdorPerception` module processes these raw values (e.g., applying a low-pass filter or adaptation logic) to produce the "perceived" odor.

#### C. Behavioral Decision

1. **State Update**: The `BehaviorAlgorithm` decides if the agent should transition between `STOP` (0) and `WALK` (1) states.
2. **Heading Update**: The algorithm calculates a new heading based on the perceived odor gradient, wind (if any), and random noise.
3. **Persistence**: If a `DirectionalPersistenceStrategy` is active, the new heading is blended with the previous heading to simulate inertia.

#### D. Kinematics & Physics

1. **Movement**: Agents update their positions (`x`, `y`) based on their current state (walking/stopped), speed, and heading.
2. **Boundary Handling**: The `Arena` checks for collisions.
    * **Walls**: Agents slide along walls or stop.
    * **Periodic Boundaries**: Agents wrap around to the other side.

### 3. Data Recording

At the end of the simulation, data is aggregated into a dictionary of NumPy arrays (time x animals) and saved to HDF5.

## Vectorization Strategy

ArthroScape is optimized for speed using **Numba**.

* **Data Structure**: Instead of a list of `Agent` objects, the simulator maintains structure-of-arrays (SoA): `self.x`, `self.y`, `self.headings` are all 1D NumPy arrays of size `N` (number of animals).
* **JIT Compilation**: Computationally intensive functions in `arena.py` (like `deposit_odor_kernels_vectorized_numba` or `get_odor_vectorized_numba`) are compiled with `@njit`.
* **Batch Operations**: The `BehaviorAlgorithm` and `Arena` methods are designed to accept and return arrays, processing all agents in a single call.

## Extending the System

* **New Behaviors**: Inherit from `BehaviorAlgorithm` and implement `update_state` and `update_heading`.
* **New Arenas**: Inherit from `GridArena` and define your own `wall_mask` or boundary logic.
* **New Sensors**: Inherit from `AgentOdorPerception` to implement complex sensory processing models.
