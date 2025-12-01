# arthroscape/sim/saver.py
"""
Data saving module for ArthroScape.

This module handles saving simulation results to HDF5 format.
It structures the data hierarchically by replicate and animal.
"""

import h5py
import numpy as np
from typing import List, Dict


def save_simulation_results_hdf5(results: List[Dict], filename: str) -> None:
    """
    Save simulation results to an HDF5 file.

    The HDF5 file structure is as follows:
    - Root
        - replicate_0
            - final_odor_grid (dataset)
            - trajectories (group)
                - animal_0 (group)
                    - x (dataset)
                    - y (dataset)
                    - heading (dataset)
                    - state (dataset)
                    - odor_left (dataset)
                    - odor_right (dataset)
                    - perceived_odor_left (dataset)
                    - perceived_odor_right (dataset)
                - animal_1 ...
            - odor_grid_history (dataset, optional)
        - replicate_1 ...

    Args:
        results (List[Dict]): A list of simulation result dictionaries.
        filename (str): The path to the output HDF5 file.
    """
    with h5py.File(filename, "w") as hf:
        for rep_idx, result in enumerate(results):
            rep_group = hf.create_group(f"replicate_{rep_idx}")
            # Save final odor grid.
            final_grid = result.get("final_odor_grid", np.array([]))
            rep_group.create_dataset("final_odor_grid", data=final_grid)

            # Save trajectories.
            traj_group = rep_group.create_group("trajectories")
            trajectories = result.get("trajectories", [])
            for animal_idx, traj in enumerate(trajectories):
                animal_group = traj_group.create_group(f"animal_{animal_idx}")
                animal_group.create_dataset("x", data=np.array(traj.get("x", [])))
                animal_group.create_dataset("y", data=np.array(traj.get("y", [])))
                animal_group.create_dataset(
                    "heading", data=np.array(traj.get("heading", []))
                )
                animal_group.create_dataset(
                    "state", data=np.array(traj.get("state", []))
                )
                animal_group.create_dataset(
                    "odor_left", data=np.array(traj.get("odor_left", []))
                )
                animal_group.create_dataset(
                    "odor_right", data=np.array(traj.get("odor_right", []))
                )
                animal_group.create_dataset(
                    "perceived_odor_left",
                    data=np.array(traj.get("perceived_odor_left", [])),
                )
                animal_group.create_dataset(
                    "perceived_odor_right",
                    data=np.array(traj.get("perceived_odor_right", [])),
                )

            # Optionally, save odor grid history if it exists.
            if "odor_grid_history" in result:
                history = np.array(result["odor_grid_history"])
                rep_group.create_dataset("odor_grid_history", data=history)
