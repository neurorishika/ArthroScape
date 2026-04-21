<div align="center">

<img src="https://github.com/user-attachments/assets/ced9c50a-e1db-4963-9bd1-6f9ff983c9f1" alt="ArthroScape Logo" width="560"/>

### *A General-Purpose Agent-Based Simulator for Arthropod Navigation & Collective Behavior*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Managed%20with-Poetry-60A5FA?style=for-the-badge&logo=python&logoColor=white)](https://python-poetry.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Numba](https://img.shields.io/badge/Accelerated%20with-Numba-00A3E0?style=for-the-badge&logo=numpy&logoColor=white)](https://numba.pydata.org/)

[🐛 Report a Bug](https://github.com/neurorishika/ArthroScape/issues) · [💡 Request Feature](https://github.com/neurorishika/ArthroScape/issues)

</div>

---

## 📋 Overview

**ArthroScape** is a high-performance, agent-based simulation framework for modeling the navigation and collective behavior of arthropods — including ants, flies, and other insects. It is built around a **pluggable behavior architecture**: any locomotion strategy can be implemented by subclassing the `BehaviorAlgorithm` abstract base class, making it straightforward to model a wide range of sensorimotor rules and navigation algorithms.

Out of the box, ArthroScape ships with a biologically inspired **osmotropotaxis** behavior as the default, which supports pheromone trail dynamics with Gaussian diffusion and scales to thousands of simultaneous agents via Numba-accelerated vectorized execution.

The framework enables exploration of how individual-level sensorimotor rules give rise to emergent group-level phenomena such as collective foraging trails, aggregation, and more.

<div align="center">
<img src="https://github.com/user-attachments/assets/ee46a4bd-f8d4-481c-b28d-e47c91ef428b" alt="ArthroScape Overview Figure" width="720"/>
<br><sub><i>The default osmotropotaxis behavior (left) — one of many pluggable behaviors — and emergent collective behavior regimes (right): individual foraging, collective foraging, and aggregation arise from varying pheromone sensitivity and diffusion width.</i></sub>
</div>

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🐜 **Pluggable Behavior System** | Implement any locomotion strategy by subclassing `BehaviorAlgorithm`; osmotropotaxis included as the default |
| 🌿 **Pheromone Trail Dynamics** | Trail deposition with Gaussian diffusion; tunable diffusion width and sensitivity |
| ⚡ **Numba Acceleration** | Vectorized simulation of thousands of agents with JIT-compiled loops |
| 🏟️ **Flexible Arenas** | Circular arenas, periodic square arenas, and custom boundary conditions |
| 🎞️ **Rich Visualization** | Built-in pipeline for trail plots, density maps, and video output |
| 💾 **HDF5 Output** | Structured, compressed result storage for large-scale parameter sweeps |
| 🔀 **Parallel Runs** | Multi-replicate sweeps via `joblib` parallel execution |
| 🖼️ **Image / Video Odor Sources** | Load static or dynamic odor landscapes from image/video files |

---

## 🔬 Example Behavior: Osmotropotaxis (Default)

ArthroScape ships with a default behavior that implements **osmotropotaxis** — a biologically grounded navigation strategy where agents steer by comparing odor concentrations at their left and right sensors. This is one example of the behavior algorithms you can plug into the framework; any custom `BehaviorAlgorithm` subclass can replace it.

In the default behavior, at each timestep an agent can:

- **Stop** (at rate λ_stop) or **Walk** (at rate λ_walk)
- **Turn** with rate λ_turn + |O_L − O_R| · sensitivity
  - Turn **left** if O_L − O_R > 0
  - Turn **right** if O_L − O_R < 0
  - Turn in a **random** direction if O_L − O_R = 0

Pheromone is released continuously from the agent's midpoint and spreads as a Gaussian kernel. By varying *sensitivity* and *diffusion width*, the model reproduces a continuum from **individual foraging** → **collective foraging** → **aggregation**.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 – 3.12
- [Poetry](https://python-poetry.org/) package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/neurorishika/ArthroScape.git
cd ArthroScape

# 2. Install dependencies with Poetry
poetry install

# 3. Activate the virtual environment
poetry shell
```

### Running a Simulation

```bash
# Run with default settings (periodic square arena, 100 agents, 10 replicates)
poetry run python -m arthroscape.sim.main

# Run with a circular arena and custom parameters
poetry run python -m arthroscape.sim.main \
    --arena circular \
    --n_animals 500 \
    --n_replicates 5 \
    --sensitivity 1.0 \
    --diffusion_width 5.0 \
    --output data/my_simulation/results.h5
```

### Jupyter Notebooks

```bash
poetry run jupyter notebook
```

---

## 📁 Project Structure

```
ArthroScape/
├── arthroscape/               # Main Python package
│   └── sim/                   # Core simulation engine
│       ├── main.py            # CLI entry point
│       ├── simulator.py       # Agent simulation loop
│       ├── runner.py          # Sequential & vectorized runners
│       ├── arena.py           # Arena boundary definitions
│       ├── behavior.py        # Locomotion / turning behavior
│       ├── odor_release.py    # Trail deposition strategies
│       ├── odor_perception.py # Bilateral odor sensing
│       ├── odor_sources.py    # Image / video odor landscapes
│       ├── config.py          # Simulation configuration dataclass
│       ├── visualization.py   # Plotting & video pipeline
│       └── saver.py           # HDF5 result serialization
├── analysis/                  # Analysis notebooks & scripts
├── data/                      # Raw simulation outputs
├── processed_data/            # Post-processed results
├── scripts/                   # Utility scripts
├── tests/                     # Unit tests
├── utils/                     # Build and update utilities
│   ├── build.py
│   ├── update.py
│   └── quickstart.py
├── docs/                      # Documentation source
├── pyproject.toml             # Poetry project configuration
└── README.md
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**ArthroScape** was designed and developed by [Rishika Mohanta](https://neurorishika.github.io/).

---

<div align="center">
<sub>Built with ❤️ for arthropod behavioral neuroscience · <a href="https://neurorishika.github.io/">neurorishika</a></sub>
</div>
