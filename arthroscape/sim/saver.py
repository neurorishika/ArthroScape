# arthroscape/sim/saver.py
import h5py
import numpy as np
from typing import List, Dict

def save_simulation_results_hdf5(results: List[Dict], filename: str) -> None:
    """
    Save simulation results to an HDF5 file.

    Each replicate is stored as a group named 'replicate_i'.
    For each replicate:
      - 'final_odor_grid' dataset holds the final odor grid.
      - 'trajectories' group holds per-animal data, with each animal stored as group 'animal_j'
        containing datasets:
          'x', 'y', 'heading', 'state', 'odor_left', 'odor_right'.
      - Optionally, if the result includes 'odor_grid_history', it is saved as a dataset.
    """
    with h5py.File(filename, 'w') as hf:
        for rep_idx, result in enumerate(results):
            rep_group = hf.create_group(f'replicate_{rep_idx}')
            # Save final odor grid.
            final_grid = result.get('final_odor_grid', np.array([]))
            rep_group.create_dataset('final_odor_grid', data=final_grid)
            
            # Save trajectories.
            traj_group = rep_group.create_group('trajectories')
            trajectories = result.get('trajectories', [])
            for animal_idx, traj in enumerate(trajectories):
                animal_group = traj_group.create_group(f'animal_{animal_idx}')
                animal_group.create_dataset('x', data=np.array(traj.get('x', [])))
                animal_group.create_dataset('y', data=np.array(traj.get('y', [])))
                animal_group.create_dataset('heading', data=np.array(traj.get('heading', [])))
                animal_group.create_dataset('state', data=np.array(traj.get('state', [])))
                animal_group.create_dataset('odor_left', data=np.array(traj.get('odor_left', [])))
                animal_group.create_dataset('odor_right', data=np.array(traj.get('odor_right', [])))
                animal_group.create_dataset('perceived_odor_left', data=np.array(traj.get('perceived_odor_left', [])))
                animal_group.create_dataset('perceived_odor_right', data=np.array(traj.get('perceived_odor_right', [])))
            
            # Optionally, save odor grid history if it exists.
            if 'odor_grid_history' in result:
                history = np.array(result['odor_grid_history'])
                rep_group.create_dataset('odor_grid_history', data=history)
