# Visualizing Results

ArthroScape includes a visualization pipeline to help you inspect simulation outcomes.

## The Visualization Pipeline

The `VisualizationPipeline` class in `arthroscape.sim.visualization` handles plotting. It takes the simulation results, configuration, and arena object as input.

```python
from arthroscape.sim.visualization import VisualizationPipeline

# Assuming you have 'results', 'config', and 'arena'
pipeline = VisualizationPipeline(results, config, arena)
```

## Common Plots

### Trajectories

Plot the paths of all agents, overlaid on the final odor landscape.

```python
pipeline.plot_trajectories_with_odor(
    sim_index=0,        # Index of the replicate to plot
    show=True,          # Display the plot
    save_path="traj.png" # Optional: save to file
)
```

### Odor Grid

Visualize the final state of the odor concentration grid.

```python
pipeline.plot_final_odor_grid(
    sim_index=0,
    show=True
)
```

### Odor Time Series

Plot the average odor concentration experienced by agents over time.

```python
pipeline.plot_odor_time_series(
    sim_index=0,
    show=True
)
```

## Creating Animations

You can create a video animation of the simulation. This requires `opencv-python`.

```python
pipeline.animate_enhanced_trajectory_opencv(
    sim_index=0,
    fps=60,
    output_file="simulation_video.mp4",
    display=True
)
```

## Custom Plotting

The simulation results are standard NumPy arrays, so you can also use Matplotlib directly.

```python
import matplotlib.pyplot as plt

# Access data for the first replicate
# Shape: (time_steps, n_animals)
x_coords = results[0]['x']
y_coords = results[0]['y']

plt.figure()
plt.plot(x_coords, y_coords, alpha=0.5)
plt.title("Agent Trajectories")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.show()
```
