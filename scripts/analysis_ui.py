import os
import tkinter as tk
from tkinter import filedialog, messagebox
from arthroscape.sim.config import SimulationConfig
from arthroscape.sim.arena import GridArena
from arthroscape.sim.visualization import VisualizationPipeline
import h5py

# Default data directory
default_data_dir = os.path.abspath("./data")


def load_simulation_results_hdf5(filename):
    """
    Load simulation results from an HDF5 file saved by the simulation saver.
    Returns a list of dictionaries, one per replicate.
    """
    results = []
    with h5py.File(filename, 'r') as hf:
        for rep_key in hf.keys():
            rep = hf[rep_key]
            rep_result = {}
            # Load final odor grid
            rep_result['final_odor_grid'] = rep['final_odor_grid'][()]
            # Load trajectories (each animal)
            traj_group = rep['trajectories']
            trajectories = []
            for animal_key in traj_group.keys():
                animal = traj_group[animal_key]
                traj = {
                    'x': animal['x'][()].tolist(),
                    'y': animal['y'][()].tolist(),
                    'heading': animal['heading'][()].tolist(),
                    'state': animal['state'][()].tolist(),
                    'odor_left': animal['odor_left'][()].tolist(),
                    'odor_right': animal['odor_right'][()].tolist()
                }
                if 'perceived_odor_left' in animal:
                    traj['perceived_odor_left'] = animal['perceived_odor_left'][()].tolist()
                if 'perceived_odor_right' in animal:
                    traj['perceived_odor_right'] = animal['perceived_odor_right'][()].tolist()
                trajectories.append(traj)
            rep_result['trajectories'] = trajectories
            results.append(rep_result)
    return results

# Main analysis function
def run_analysis(sim_file, output_folder, options):
    sim_results = load_simulation_results_hdf5(sim_file)
    config = SimulationConfig()
    config.__post_init__()

    arena = GridArena(config.grid_x_min, config.grid_x_max,
                      config.grid_y_min, config.grid_y_max,
                      config.grid_resolution, config=config)

    viz = VisualizationPipeline(sim_results=sim_results, config=config, arena=arena)

    for rep_index in range(len(sim_results)):
        rep_out = os.path.join(output_folder, f"replicate_{rep_index}")
        os.makedirs(rep_out, exist_ok=True)

        if options['Trajectories']:
            traj_path = os.path.join(rep_out, "trajectories.png")
            viz.plot_trajectories_with_odor(
                sim_index=rep_index, save_path=traj_path,
                wraparound=options['Wraparound'], show=False
            )

        if options['Odor Grid']:
            grid_path = os.path.join(rep_out, "final_odor_grid.png")
            viz.plot_final_odor_grid(sim_index=rep_index, save_path=grid_path, show=False)

        if options['Odor Time Series']:
            ts_path = os.path.join(rep_out, "odor_time_series.png")
            viz.plot_odor_time_series(sim_index=rep_index, save_path=ts_path, show=False)

        if options['Animation']:
            anim_path = os.path.join(rep_out, "trajectory_animation.mp4")
            viz.animate_enhanced_trajectory_opencv(
                sim_index=rep_index,
                fps=options['FPS'],
                frame_skip=options['Frame Skip'],
                output_file=anim_path,
                wraparound=options['Wraparound'],
                display=True, progress=True
            )

    messagebox.showinfo("Analysis Complete", "Selected analyses have been completed!")

# UI Functions
def select_file():
    file_path = filedialog.askopenfilename(initialdir=default_data_dir, filetypes=[("HDF5 Files", "*.h5")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def select_folder():
    folder_path = filedialog.askdirectory(initialdir=default_data_dir)
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, folder_path)

def start_analysis():
    sim_file = file_entry.get()
    output_folder = folder_entry.get()

    try:
        fps = int(fps_entry.get())
        frame_skip = int(frame_skip_entry.get())
    except ValueError:
        messagebox.showerror("Invalid Input", "FPS and Frame Skip must be integers.")
        return

    options = {
        'Trajectories': traj_var.get(),
        'Odor Grid': odor_var.get(),
        'Odor Time Series': time_var.get(),
        'Animation': anim_var.get(),
        'FPS': fps,
        'Frame Skip': frame_skip,
        'Wraparound': wrap_var.get()
    }

    if not os.path.isfile(sim_file) or not os.path.isdir(output_folder):
        messagebox.showerror("Path Error", "Invalid file or output directory.")
        return

    run_analysis(sim_file, output_folder, options)

# UI Creation
root = tk.Tk()
root.title("Simulation Analysis")

# File Selection
tk.Label(root, text="Simulation File (.h5):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
file_entry = tk.Entry(root, width=40)
file_entry.grid(row=0, column=1, padx=5)
tk.Button(root, text="Browse", command=select_file).grid(row=0, column=2)

# Folder Selection
tk.Label(root, text="Output Folder:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
folder_entry = tk.Entry(root, width=40)
folder_entry.grid(row=1, column=1, padx=5)
tk.Button(root, text="Browse", command=select_folder).grid(row=1, column=2)

# Analysis Options
traj_var = tk.BooleanVar(value=True)
odor_var = tk.BooleanVar(value=True)
time_var = tk.BooleanVar(value=False)
anim_var = tk.BooleanVar(value=True)
wrap_var = tk.BooleanVar(value=True)

options_frame = tk.LabelFrame(root, text="Analyses")
options_frame.grid(row=2, column=0, columnspan=3, pady=10, padx=5)
tk.Checkbutton(options_frame, text="Trajectories Plot", variable=traj_var).grid(row=0, column=0, sticky='w')
tk.Checkbutton(options_frame, text="Odor Grid", variable=odor_var).grid(row=1, column=0, sticky='w')
tk.Checkbutton(options_frame, text="Odor Time Series", variable=time_var).grid(row=2, column=0, sticky='w')
tk.Checkbutton(options_frame, text="Animation Video", variable=anim_var).grid(row=3, column=0, sticky='w')
tk.Checkbutton(options_frame, text="Wraparound (PBC Arena)", variable=wrap_var).grid(row=4, column=0, sticky='w')

# Animation parameters
params_frame = tk.LabelFrame(root, text="Animation Parameters")
params_frame.grid(row=3, column=0, columnspan=3, pady=10, padx=5)

tk.Label(params_frame, text="FPS:").grid(row=0, column=0, sticky='e')
fps_entry = tk.Entry(params_frame, width=10)
fps_entry.insert(0, "30")
fps_entry.grid(row=0, column=1)

tk.Label(params_frame, text="Frame Skip:").grid(row=1, column=0, sticky='e')
frame_skip_entry = tk.Entry(params_frame, width=10)
frame_skip_entry.insert(0, "30")
frame_skip_entry.grid(row=1, column=1)

# Action Buttons
tk.Button(root, text="Run", command=start_analysis, bg="green", fg="white").grid(row=4, column=1, pady=10)
tk.Button(root, text="Quit", command=root.quit, bg="red", fg="white").grid(row=4, column=2, pady=10)

root.mainloop()
