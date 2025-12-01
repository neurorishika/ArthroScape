# Running Simulations in Parallel

ArthroScape supports parallel execution of simulation replicates to speed up data collection.

## Using the CLI

To run simulations in parallel using the command line, simply add the `--parallel` flag.

```bash
python -m arthroscape.sim.main --arena circular --replicates 10 --parallel
```

This will use all available CPU cores to run the 10 replicates concurrently.

## Using the Python API

When using the Python API, set the `parallel` argument to `True` in the runner functions.

```python
from arthroscape.sim.runner import run_simulations_vectorized

results = run_simulations_vectorized(
    config, 
    behavior, 
    arena, 
    odor_release, 
    n_replicates=20, 
    parallel=True
)
```

## Performance Considerations

* **Vectorization vs. Parallelization**: ArthroScape uses Numba for vectorization within a single simulation (handling multiple agents). Parallelization runs multiple *replicates* (independent simulations) at the same time. Using both is highly recommended for large-scale parameter sweeps.
* **Memory Usage**: Each parallel process requires its own memory. If you are simulating very large arenas or recording high-resolution odor history (`record_odor_history=True`), be mindful of your system's RAM.
* **CPU Cores**: By default, `joblib` (used internally) attempts to use all available cores.
