# arthroscape/sim/runner.py
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any
import numpy as np
from .config import SimulationConfig
from .behavior import BehaviorAlgorithm
from .arena import Arena
from .odor_release import OdorReleaseStrategy
from .simulator import MultiAnimalSimulator

def simulate_replicate(config: SimulationConfig, behavior: BehaviorAlgorithm, arena: Arena,
                       odor_release_strategy: OdorReleaseStrategy, seed: int = None) -> Dict[str, Any]:
    simulator = MultiAnimalSimulator(config, behavior, arena, odor_release_strategy, seed)
    result = simulator.simulate()
    result["seed"] = seed
    return result

def run_simulations(config: SimulationConfig, behavior: BehaviorAlgorithm, arena: Arena,
                    odor_release_strategy: OdorReleaseStrategy, n_replicates: int = 1,
                    parallel: bool = True) -> List[Dict[str, Any]]:
    results = []
    seeds = [np.random.SeedSequence().entropy for _ in range(n_replicates)]
    if parallel:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(simulate_replicate, config, behavior, arena, odor_release_strategy, seed): i for i, seed in enumerate(seeds)}
            for future in tqdm(as_completed(futures), total=n_replicates, desc="Simulating"):
                results.append(future.result())
    else:
        for seed in tqdm(seeds, desc="Simulating"):
            results.append(simulate_replicate(config, behavior, arena, odor_release_strategy, seed))
    return results

def save_simulation_results(results: List[Dict[str, Any]], filename: str) -> None:
    np.savez_compressed(filename, results=results)
