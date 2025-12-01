# Analyzing Simulation Data

ArthroScape saves simulation results in HDF5 format, which is efficient for large datasets. This guide explains how to load and analyze this data.

## Loading Results

You can use the `h5py` library or ArthroScape's built-in loader (if available, otherwise standard `h5py` is recommended).

```python
import h5py
import numpy as np

filename = "path/to/results.h5"

with h5py.File(filename, "r") as f:
    # List all replicates
    replicates = list(f.keys())
    print(f"Found {len(replicates)} replicates: {replicates}")
    
    # Access data for the first replicate
    rep0 = f['replicate_0']
    
    # Load datasets into NumPy arrays
    x = rep0['x'][:]          # Shape: (time_steps, n_animals)
    y = rep0['y'][:]
    headings = rep0['headings'][:]
    odor = rep0['odor'][:]    # Odor experienced by agents
```

## Data Structure

Each replicate group in the HDF5 file typically contains:

* `x`: X-coordinates of agents over time.
* `y`: Y-coordinates of agents over time.
* `headings`: Heading angles (radians) over time.
* `odor`: Odor concentration at the agent's location (or sensors).
* `state`: Behavioral state (e.g., 0=stop, 1=walk).
* `odor_grid_history`: (Optional) Snapshots of the full odor grid if `record_odor_history` was enabled.

## Example Analysis: Success Rate

Let's calculate the fraction of agents that reached a target zone (e.g., within 10mm of the center).

```python
target_radius = 10.0
success_count = 0
total_animals = x.shape[1]

# Check final positions
final_x = x[-1, :]
final_y = y[-1, :]
distances = np.sqrt(final_x**2 + final_y**2)

successes = distances < target_radius
success_rate = np.sum(successes) / total_animals

print(f"Success Rate: {success_rate * 100:.1f}%")
```

## Example Analysis: Occupancy Map

Create a 2D histogram of agent positions to see where they spent the most time.

```python
import matplotlib.pyplot as plt

# Flatten all time points and animals
all_x = x.flatten()
all_y = y.flatten()

plt.figure(figsize=(6, 6))
plt.hist2d(all_x, all_y, bins=50, cmap='hot', range=[[-100, 100], [-100, 100]])
plt.colorbar(label='Occupancy')
plt.title("Agent Occupancy Map")
plt.show()
```
