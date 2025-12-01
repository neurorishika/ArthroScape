# Using External Odor Sources

ArthroScape supports loading odor landscapes from external sources like images and videos. This enables you to:

* Use experimentally recorded odor plumes as simulation input.
* Create complex spatial patterns without code.
* Replay time-varying odor dynamics from video recordings.

## Static Images as Odor Maps

The simplest use case is loading a grayscale image where pixel brightness represents odor concentration.

### Basic Usage

```python
from arthroscape.sim.config import SimulationConfig
from arthroscape.sim.arena import PeriodicSquareArena
from arthroscape.sim.odor_sources import load_odor_from_image

# Setup
config = SimulationConfig()
arena = PeriodicSquareArena(
    config.grid_x_min, config.grid_x_max,
    config.grid_y_min, config.grid_y_max,
    config.grid_resolution, config=config
)

# Load odor landscape from image
source = load_odor_from_image(
    "my_odor_gradient.png",
    arena,
    scale=10.0,  # Scale pixel values (0-1) to odor units
    invert=False  # Set True if dark = high odor
)
```

### Advanced Image Options

For more control, use the `ImageOdorSource` class directly:

```python
from arthroscape.sim.odor_sources import ImageOdorSource

# Custom normalization function
def log_normalize(img):
    return np.log1p(img) / np.log1p(255)

source = ImageOdorSource(
    "plume_image.tiff",
    invert=True,            # Invert values
    normalize_func=log_normalize,  # Custom normalization
    channel=0               # Use only red channel (0=R, 1=G, 2=B)
)

# Apply to arena with different modes
source.apply_to_arena(arena, mode="replace", scale=5.0, offset=0.1)
```

### Application Modes

| Mode | Effect |
|:---|:---|
| `"replace"` | Overwrite the arena's odor grid entirely |
| `"add"` | Add image values to existing odor |
| `"multiply"` | Multiply existing odor by image values |
| `"max"` | Take element-wise maximum |

## Dynamic Video as Odor Field

For time-varying odor fields, use `VideoOdorSource` to stream video frames.

### Basic Video Usage

```python
from arthroscape.sim.odor_sources import VideoOdorSource

# Load video
video = VideoOdorSource(
    "plume_recording.mp4",
    loop=True,       # Loop when video ends
    invert=False
)

# Apply first frame
video.apply_to_arena(arena, scale=2.0)

# In your simulation loop:
for frame in range(total_frames):
    # Advance video and apply new frame
    video.advance_frame()
    video.apply_to_arena(arena, mode="replace", scale=2.0)
    
    # ... rest of simulation step ...
```

### Video Synchronization

Use `VideoOdorReleaseStrategy` for automatic frame synchronization:

```python
from arthroscape.sim.odor_sources import VideoOdorReleaseStrategy

# Create synchronized video strategy
video_strategy = VideoOdorReleaseStrategy(
    "odor_plume.mp4",
    arena,
    mode="replace",
    scale=5.0,
    sync_mode="simulation_fps",  # Map simulation time to video time
    simulation_fps=60.0,
    loop=True
)

# In simulation loop:
for sim_frame in range(config.total_frames):
    video_strategy.update(arena, sim_frame)
    # ... rest of simulation ...
```

### Synchronization Modes

| Mode | Description |
|:---|:---|
| `"one_to_one"` | Each simulation frame = one video frame |
| `"video_fps"` | Use video's native frame rate |
| `"simulation_fps"` | Map simulation time to video time (recommended) |

### Video Options

```python
video = VideoOdorSource(
    "recording.mp4",
    loop=True,          # Loop video
    start_frame=100,    # Start at frame 100
    end_frame=500,      # End at frame 500
    frame_step=2,       # Use every 2nd frame
    preload=True,       # Load all frames into RAM (faster, more memory)
    channel=1           # Use green channel only
)
```

## Complete Example: Image + Agent Pheromone

Combine a static background odor with agent-deposited pheromone:

```python
from arthroscape.sim.config import SimulationConfig
from arthroscape.sim.arena import PeriodicSquareArena
from arthroscape.sim.behavior import DefaultBehavior
from arthroscape.sim.odor_release import ConstantOdorRelease
from arthroscape.sim.odor_sources import load_odor_from_image
from arthroscape.sim.runner import run_simulations_vectorized

# Configuration
config = SimulationConfig(
    T=300,
    number_of_animals=5,
    diffusion_coefficient=0.5,  # Enable diffusion
    odor_decay_tau=60.0         # Odor decays over 60 seconds
)

# Create arena
arena = PeriodicSquareArena(
    config.grid_x_min, config.grid_x_max,
    config.grid_y_min, config.grid_y_max,
    config.grid_resolution, config=config
)

# Load background odor from image
load_odor_from_image("food_source.png", arena, scale=2.0)

# Agents also release pheromone
odor_release = ConstantOdorRelease(config, deposit_amount=0.1)

# Run simulation
behavior = DefaultBehavior()
results = run_simulations_vectorized(
    config, behavior, arena, odor_release, n_replicates=1
)
```

## Complete Example: Video-Driven Simulation

Run agents in a dynamic, video-based odor environment:

```python
from arthroscape.sim.config import SimulationConfig
from arthroscape.sim.arena import PeriodicSquareArena
from arthroscape.sim.behavior import DefaultBehavior
from arthroscape.sim.odor_release import DefaultOdorRelease
from arthroscape.sim.odor_sources import VideoOdorReleaseStrategy
from arthroscape.sim.simulator import MultiAnimalSimulator

# Configuration
config = SimulationConfig(T=60, number_of_animals=10, fps=30)

# Create arena
arena = PeriodicSquareArena(
    config.grid_x_min, config.grid_x_max,
    config.grid_y_min, config.grid_y_max,
    config.grid_resolution, config=config
)

# Video-based dynamic odor
video_strategy = VideoOdorReleaseStrategy(
    "turbulent_plume.mp4",
    arena,
    scale=3.0,
    sync_mode="simulation_fps",
    simulation_fps=config.fps,
    loop=True
)

# No agent-based odor release
odor_release = DefaultOdorRelease()

# Create simulator
simulator = MultiAnimalSimulator(
    config=config,
    behavior=DefaultBehavior(),
    arena=arena,
    odor_release_strategy=odor_release
)

# Custom simulation loop with video updates
for frame in range(config.total_frames):
    # Update odor from video BEFORE the simulation step
    video_strategy.update(arena, frame)
    
    # ... manual step logic or use simulator internals ...

# Clean up
video_strategy.video_source.close()
```

## Utility Functions

### Create Simple Gradients

For quick testing, use the built-in gradient generator:

```python
from arthroscape.sim.odor_sources import create_gradient_odor_map

# Horizontal gradient (left to right)
horizontal = create_gradient_odor_map(
    (arena.ny, arena.nx),
    direction="horizontal",
    min_val=0.0,
    max_val=5.0
)
arena.odor_grid = horizontal

# Radial gradient (center = high)
radial = create_gradient_odor_map(
    (arena.ny, arena.nx),
    direction="radial_inward",
    min_val=0.0,
    max_val=10.0
)
arena.odor_grid = radial
```

## Tips and Best Practices

1. **Image Resolution**: Use images that roughly match your arena's grid resolution. Very high-resolution images will be downsampled.

2. **Video Performance**: For long videos, set `preload=False` (default) to stream frames. Use `preload=True` only for short videos when you need maximum speed.

3. **Coordinate System**: Images are loaded with the origin at the top-left. The arena's origin is at `(x_min, y_min)`. The image is automatically flipped if needed.

4. **Normalization**: By default, pixel values are normalized to [0, 1]. Use `scale` to convert to your desired odor units.

5. **Combining Sources**: You can layer multiple sources using `mode="add"` or `mode="max"`.
