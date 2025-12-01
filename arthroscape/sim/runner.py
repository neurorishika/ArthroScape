# arthroscape/sim/runner.py
"""
Simulation runner module for ArthroScape.

This module provides functions to run simulations in parallel or sequentially.
It handles the execution of multiple replicates and the collection of results.
"""

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


def simulate_replicate(
    config: SimulationConfig,
    behavior: BehaviorAlgorithm,
    arena: Arena,
    odor_release_strategy: OdorReleaseStrategy,
    seed: int = None,
) -> Dict[str, Any]:
    """
    Run a single simulation replicate.

    Args:
        config (SimulationConfig): Simulation configuration.
        behavior (BehaviorAlgorithm): Behavior algorithm instance.
        arena (Arena): Arena instance.
        odor_release_strategy (OdorReleaseStrategy): Odor release strategy instance.
        seed (int, optional): Random seed for this replicate.

    Returns:
        Dict[str, Any]: Simulation results dictionary containing tracks, odor fields, etc.
    """
    simulator = MultiAnimalSimulator(
        config, behavior, arena, odor_release_strategy, seed
    )
    result = simulator.simulate()
    result["seed"] = seed
    return result


def run_simulations(
    config: SimulationConfig,
    behavior: BehaviorAlgorithm,
    arena: Arena,
    odor_release_strategy: OdorReleaseStrategy,
    n_replicates: int = 1,
    parallel: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run multiple simulation replicates, optionally in parallel.

    Args:
        config (SimulationConfig): Simulation configuration.
        behavior (BehaviorAlgorithm): Behavior algorithm instance.
        arena (Arena): Arena instance.
        odor_release_strategy (OdorReleaseStrategy): Odor release strategy instance.
        n_replicates (int): Number of replicates to run. Defaults to 1.
        parallel (bool): Whether to run replicates in parallel using ProcessPoolExecutor. Defaults to True.

    Returns:
        List[Dict[str, Any]]: A list of result dictionaries, one for each replicate.
    """
    results = []
    seeds = [np.random.SeedSequence().entropy for _ in range(n_replicates)]
    if parallel:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    simulate_replicate,
                    config,
                    behavior,
                    arena,
                    odor_release_strategy,
                    seed,
                ): i
                for i, seed in enumerate(seeds)
            }
            for future in tqdm(
                as_completed(futures), total=n_replicates, desc="Simulating"
            ):
                results.append(future.result())
    else:
        for seed in tqdm(seeds, desc="Simulating"):
            results.append(
                simulate_replicate(config, behavior, arena, odor_release_strategy, seed)
            )
    return results


def save_simulation_results(results: List[Dict[str, Any]], filename: str) -> None:
    """
    Save simulation results to a compressed NumPy file.

    Args:
        results (List[Dict[str, Any]]): List of simulation result dictionaries.
        filename (str): Path to the output file (should end in .npz).
    """
    np.savez_compressed(filename, results=results)


def simulate_replicate_vectorized(
    config: SimulationConfig,
    behavior: BehaviorAlgorithm,
    arena: Arena,
    odor_release_strategy: OdorReleaseStrategy,
    seed: int = None,
) -> Dict[str, Any]:
    """
    Run a single vectorized simulation replicate.

    Args:
        config (SimulationConfig): Simulation configuration.
        behavior (BehaviorAlgorithm): Behavior algorithm instance.
        arena (Arena): Arena instance.
        odor_release_strategy (OdorReleaseStrategy): Odor release strategy instance.
        seed (int, optional): Random seed for this replicate.

    Returns:
        Dict[str, Any]: Simulation results dictionary.
    """
    simulator = MultiAnimalSimulator(
        config, behavior, arena, odor_release_strategy, seed
    )
    result = simulator.simulate_vectorized()
    result["seed"] = seed
    return result


def run_simulations_vectorized(
    config: SimulationConfig,
    behavior: BehaviorAlgorithm,
    arena: Arena,
    odor_release_strategy: OdorReleaseStrategy,
    n_replicates: int = 1,
    parallel: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run multiple vectorized simulation replicates, optionally in parallel.

    Args:
       config (SimulationConfig): Simulation configuration.
       behavior (BehaviorAlgorithm): Behavior algorithm instance.
       arena (Arena): Arena instance.
       odor_release_strategy (OdorReleaseStrategy): Odor release strategy instance.
       n_replicates (int): Number of replicates to run. Defaults to 1.
       parallel (bool): Whether to run replicates in parallel. Defaults to True.

    Returns:
       List[Dict[str, Any]]: A list of result dictionaries.
    """
    results = []
    seeds = [np.random.SeedSequence().entropy for _ in range(n_replicates)]
    if parallel:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    simulate_replicate_vectorized,
                    config,
                    behavior,
                    arena,
                    odor_release_strategy,
                    seed,
                ): i
                for i, seed in enumerate(seeds)
            }
            for future in tqdm(
                as_completed(futures), total=n_replicates, desc="Simulating"
            ):
                results.append(future.result())
    else:
        for seed in tqdm(seeds, desc="Simulating"):
            results.append(
                simulate_replicate_vectorized(
                    config, behavior, arena, odor_release_strategy, seed
                )
            )
    return results
