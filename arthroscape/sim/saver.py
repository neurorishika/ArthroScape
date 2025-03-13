# arthroscape/sim/saver.py
import h5py
import numpy as np
from typing import List, Dict

def save_simulation_results_hdf5(results: List[Dict], filename: str) -> None:
    """
    Save simulation results to an HDF5 file.
    Each replicate is stored as a group.
    """
    with h5py.File(filename, 'w') as hf:
        for idx, result in enumerate(results):
            grp = hf.create_group(f'replicate_{idx}')
            # Save final odor grid.
            grp.create_dataset('final_odor_grid', data=result['final_odor_grid'])
            # For trajectories (multi-animal)
            traj_grp = grp.create_group('trajectories')
            for a, traj in enumerate(result['trajectories']):
                a_grp = traj_grp.create_group(f'animal_{a}')
                a_grp.create_dataset('x', data=np.array(traj['x']))
                a_grp.create_dataset('y', data=np.array(traj['y']))
                a_grp.create_dataset('heading', data=np.array(traj['heading']))
                a_grp.create_dataset('state', data=np.array(traj['state']))
                a_grp.create_dataset('odor_left', data=np.array(traj['odor_left']))
                a_grp.create_dataset('odor_right', data=np.array(traj['odor_right']))
