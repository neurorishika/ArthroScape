# arthroscape/sim/main.py
import argparse
import os
import logging
from datetime import datetime

from arthroscape.sim.config import SimulationConfig
from arthroscape.sim.arena import create_circular_arena_with_annular_trail, PeriodicSquareArena
from arthroscape.sim.behavior import DefaultBehavior
from arthroscape.sim.odor_release import DefaultOdorRelease, ConstantOdorRelease
from arthroscape.sim.runner import run_simulations
from arthroscape.sim.visualization import VisualizationPipeline
from arthroscape.sim.saver import save_simulation_results_hdf5

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_default_save_path(args) -> str:
    """Generate a default HDF5 file path under data/simulation/MM-DD-YY/{time}_{description}/results.h5"""
    date_str = datetime.now().strftime("%m-%d-%Y")
    time_str = datetime.now().strftime("%H_%M_%S")
    description = f"{time_str}_{args.arena}_arena_{args.animals}x{args.replicates}_{args.odor_release}"
    dir_path = os.path.join("data", "simulation", date_str, description)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, "results.h5")

def get_default_plots_path(args, save_path: str) -> str:
    """Generate a default directory for saving plots as a subdirectory 'plots' next to the HDF5 file."""
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(save_path)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def main():
    parser = argparse.ArgumentParser(description="Multi-Animal Fly Simulation in a Circular Arena")
    parser.add_argument("--arena", type=str, default="circular", choices=["circular", "pbc"],
                        help="Arena type: 'circular' or 'pbc'")
    parser.add_argument("--replicates", type=int, default=1, help="Number of simulation replicates")
    parser.add_argument("--animals", type=int, default=1, help="Number of animals per replicate")
    parser.add_argument("--parallel", action="store_true", help="Run simulations in parallel")
    parser.add_argument("--odor_release", type=str, default="constant", choices=["none", "conditional", "constant"],
                        help="Odor release strategy: 'none', 'conditional', or 'constant'")
    parser.add_argument("--deposit_amount", type=float, default=0.5,
                        help="Deposit amount for constant odor release (pheromone)")
    # The --save option expects an HDF5 filename; if not provided, a default path is generated.
    parser.add_argument("--save", type=str, default="", help="Filename to save results as HDF5 (e.g., results.h5)")
    parser.add_argument("--visualize", action="store_true", help="Visualize simulation results and save plots")
    args = parser.parse_args()

    # Create simulation configuration.
    config = SimulationConfig(number_of_animals=args.animals)

    # Get the behavioral algorithm.
    behavior = DefaultBehavior()

    # Choose the arena.
    if args.arena == "circular":
        arena = create_circular_arena_with_annular_trail(config,
                                                         arena_radius=75.0,
                                                         trail_radius=42.5,
                                                         trail_width=5.0,
                                                         trail_odor=0.0)
    elif args.arena == "pbc":
        arena = PeriodicSquareArena(config.grid_x_min, config.grid_x_max,
                                    config.grid_y_min, config.grid_y_max,
                                    config.grid_resolution, config=config)
    else:
        raise ValueError("Unknown arena type selected.")

    # Choose odor release strategy.
    if args.odor_release == "none":
        odor_release_strategy = DefaultOdorRelease()
    elif args.odor_release == "constant":
        odor_release_strategy = ConstantOdorRelease(config=config,deposit_amount=args.deposit_amount)
    else:
        raise ValueError("Unknown odor release strategy selected.")

    logger.info("Starting simulation...")
    simulation_results = run_simulations(config, behavior, arena, odor_release_strategy,
                                         n_replicates=args.replicates, parallel=args.parallel)
    logger.info("Simulation complete.")

    # Determine the save path for simulation results.
    if not args.save:
        args.save = get_default_save_path(args)
        logger.info(f"Auto-generated simulation save path: {args.save}")
    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    save_simulation_results_hdf5(simulation_results, args.save)
    # copy config to the same directory
    config_path = "arthroscape/sim/config.py"
    os.system(f"cp {config_path} {os.path.dirname(os.path.abspath(args.save))}")
    logger.info(f"Simulation results saved to {args.save}")

    # If visualization is enabled, save plots for each replicate.
    if args.visualize:
        plots_dir = get_default_plots_path(args, args.save)
        # Create a VisualizationPipeline instance with all replicates.
        viz = VisualizationPipeline(sim_results=simulation_results, config=config, arena=arena)
        # For each replicate, save separate plots in a subdirectory.
        for rep_index in range(len(simulation_results)):
            rep_plots_dir = os.path.join(plots_dir, f"replicate_{rep_index}")
            os.makedirs(rep_plots_dir, exist_ok=True)
            
            # Save trajectory plot.
            traj_plot_path = os.path.join(rep_plots_dir, "trajectories.png")
            viz.plot_trajectories_with_odor(sim_index=rep_index, show=False,
                                            save_path=traj_plot_path,
                                            wraparound=True if args.arena == "pbc" else False)
            # Save final odor grid plot.
            odor_grid_path = os.path.join(rep_plots_dir, "final_odor_grid.png")
            viz.plot_final_odor_grid(show=False, save_path=odor_grid_path)
            # Save odor time series plot.
            odor_ts_path = os.path.join(rep_plots_dir, "odor_time_series.png")
            viz.plot_odor_time_series(sim_index=rep_index, show=False, save_path=odor_ts_path)
            # Save animation.
            animation_path = os.path.join(rep_plots_dir, "trajectory_animation.mp4")
            viz.animate_enhanced_trajectory_opencv(sim_index=rep_index, fps=config.fps, frame_skip=8,
                                            output_file=animation_path,
                                            wraparound=True if args.arena == "pbc" else False)
            logger.info(f"Saved plots and animation for replicate {rep_index} in {rep_plots_dir}")

if __name__ == "__main__":
    main()
