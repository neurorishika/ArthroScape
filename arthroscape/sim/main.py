# arthroscape/sim/main.py
"""
Main entry point for running ArthroScape simulations.

This module provides a command-line interface (CLI) to configure and run simulations.
It handles argument parsing, simulation setup (arena, behavior, odor release),
execution (sequential or parallel, vectorized or not), and result saving/visualization.
"""

import argparse
import os
import logging
from datetime import datetime

from arthroscape.sim.config import SimulationConfig
from arthroscape.sim.arena import (
    create_circular_arena_with_annular_trail,
    PeriodicSquareArena,
    create_pbc_arena_with_line,
)
from arthroscape.sim.behavior import DefaultBehavior
from arthroscape.sim.odor_release import DefaultOdorRelease, ConstantOdorRelease
from arthroscape.sim.odor_sources import (
    ImageOdorSource,
    VideoOdorSource,
    VideoOdorReleaseStrategy,
    load_odor_from_image,
)
from arthroscape.sim.runner import run_simulations, run_simulations_vectorized
from arthroscape.sim.visualization import VisualizationPipeline
from arthroscape.sim.saver import save_simulation_results_hdf5

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_default_save_path(args) -> str:
    """
    Generate a default HDF5 file path for saving results.

    The path is structured as: `data/simulation/MM-DD-YY/{time}_{description}/results.h5`.
    The description includes the arena type, number of animals, replicates, and odor release strategy.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        str: The absolute path to the output HDF5 file.
    """
    date_str = datetime.now().strftime("%m-%d-%Y")
    time_str = datetime.now().strftime("%H_%M_%S")

    # Build description string
    odor_source_str = args.odor_release
    if args.odor_image:
        odor_source_str += "_img"
    if args.odor_video:
        odor_source_str += "_vid"

    description = f"{time_str}_{args.arena}_arena_{args.animals}x{args.replicates}_{odor_source_str}"
    dir_path = os.path.join("data", "simulation", date_str, description)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, "results.h5")


def get_default_plots_path(args, save_path: str) -> str:
    """
    Generate a default directory for saving plots.

    Creates a 'plots' subdirectory in the same directory as the results file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        save_path (str): Path to the results HDF5 file.

    Returns:
        str: Path to the plots directory.
    """
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(save_path)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def main():
    """
    Main execution function.

    Parses command-line arguments, sets up the simulation environment, runs the simulation,
    saves the results, and optionally generates visualizations.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Animal Fly Simulation in a Circular Arena"
    )
    parser.add_argument(
        "--arena",
        type=str,
        default="circular",
        choices=["circular", "pbc", "pbc-line"],
        help="Arena type: 'circular' or 'pbc' or 'pbc-line'",
    )
    parser.add_argument(
        "--replicates", type=int, default=1, help="Number of simulation replicates"
    )
    parser.add_argument(
        "--animals", type=int, default=1, help="Number of animals per replicate"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run simulations in parallel"
    )
    parser.add_argument(
        "--odor_release",
        type=str,
        default="constant",
        choices=["none", "conditional", "constant"],
        help="Odor release strategy: 'none', 'conditional', or 'constant'",
    )
    parser.add_argument(
        "--deposit_amount",
        type=float,
        default=0.5,
        help="Deposit amount for constant odor release (pheromone)",
    )
    # The --save option expects an HDF5 filename; if not provided, a default path is generated.
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Filename to save results as HDF5 (e.g., results.h5)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize simulation results and save plots",
    )
    parser.add_argument(
        "--dont-vectorize",
        action="store_true",
        help="Run the simulation in non-vectorized mode",
    )
    # External odor source options
    parser.add_argument(
        "--odor_image",
        type=str,
        default="",
        help="Path to an image file to use as static odor landscape",
    )
    parser.add_argument(
        "--odor_video",
        type=str,
        default="",
        help="Path to a video file to use as dynamic odor field",
    )
    parser.add_argument(
        "--odor_scale",
        type=float,
        default=1.0,
        help="Scaling factor for external odor source values (default: 1.0)",
    )
    parser.add_argument(
        "--odor_invert",
        action="store_true",
        help="Invert external odor source (dark becomes high concentration)",
    )
    parser.add_argument(
        "--odor_mode",
        type=str,
        default="replace",
        choices=["replace", "add", "multiply", "max"],
        help="How to apply external odor: 'replace', 'add', 'multiply', or 'max'",
    )
    parser.add_argument(
        "--video_loop",
        action="store_true",
        help="Loop the odor video when it ends (default: no loop)",
    )
    parser.add_argument(
        "--video_sync",
        type=str,
        default="simulation_fps",
        choices=["one_to_one", "video_fps", "simulation_fps"],
        help="Video synchronization mode: 'one_to_one', 'video_fps', or 'simulation_fps'",
    )
    args = parser.parse_args()

    # Create simulation configuration.
    config = SimulationConfig(number_of_animals=args.animals)

    # Get the behavioral algorithm.
    behavior = DefaultBehavior()

    # Choose the arena.
    if args.arena == "circular":
        arena = create_circular_arena_with_annular_trail(
            config,
            arena_radius=75.0,
            trail_radius=50.0,
            trail_width=5.0,
            trail_odor=1.0,
        )
    elif args.arena == "pbc":
        arena = PeriodicSquareArena(
            config.grid_x_min,
            config.grid_x_max,
            config.grid_y_min,
            config.grid_y_max,
            config.grid_resolution,
            config=config,
        )
    elif args.arena == "pbc-line":
        arena = create_pbc_arena_with_line(
            config,
            line_width=5.0,
            line_odor=1.0,
        )
    else:
        raise ValueError("Unknown arena type selected.")

    # Apply external odor source if provided
    video_strategy = None
    if args.odor_image and args.odor_video:
        raise ValueError("Cannot use both --odor_image and --odor_video. Choose one.")

    if args.odor_image:
        if not os.path.exists(args.odor_image):
            raise FileNotFoundError(f"Odor image not found: {args.odor_image}")
        logger.info(f"Loading odor landscape from image: {args.odor_image}")
        image_source = ImageOdorSource(args.odor_image, invert=args.odor_invert)
        image_source.apply_to_arena(arena, mode=args.odor_mode, scale=args.odor_scale)
        logger.info(
            f"Applied image odor source (scale={args.odor_scale}, "
            f"mode={args.odor_mode}, invert={args.odor_invert})"
        )

    if args.odor_video:
        if not os.path.exists(args.odor_video):
            raise FileNotFoundError(f"Odor video not found: {args.odor_video}")
        logger.info(f"Loading dynamic odor field from video: {args.odor_video}")
        video_strategy = VideoOdorReleaseStrategy(
            args.odor_video,
            arena,
            mode=args.odor_mode,
            scale=args.odor_scale,
            sync_mode=args.video_sync,
            simulation_fps=config.fps,
            loop=args.video_loop,
            invert=args.odor_invert,
        )
        logger.info(
            f"Video odor source configured (scale={args.odor_scale}, "
            f"mode={args.odor_mode}, sync={args.video_sync}, loop={args.video_loop})"
        )

    # Choose odor release strategy.
    if args.odor_release == "none":
        odor_release_strategy = DefaultOdorRelease()
    elif args.odor_release == "constant":
        odor_release_strategy = ConstantOdorRelease(
            config=config, deposit_amount=args.deposit_amount
        )
    else:
        raise ValueError("Unknown odor release strategy selected.")

    logger.info("Starting simulation...")

    # Note: Video odor sources require custom simulation loops for per-frame updates
    if video_strategy is not None:
        logger.warning(
            "Video odor source detected. Note: The standard runners apply the first "
            "frame only. For dynamic video updates, use a custom simulation loop with "
            "video_strategy.update(arena, frame_index) called each frame."
        )
        # Apply the first frame of the video to the arena
        video_strategy.update(arena, 0)

    if args.dont_vectorize:
        simulation_results = run_simulations(
            config,
            behavior,
            arena,
            odor_release_strategy,
            n_replicates=args.replicates,
            parallel=args.parallel,
        )
    else:
        simulation_results = run_simulations_vectorized(
            config,
            behavior,
            arena,
            odor_release_strategy,
            n_replicates=args.replicates,
            parallel=args.parallel,
        )
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

    if args.visualize:
        logger.info("Generating visualizations...")
        plots_path = get_default_plots_path(args, args.save)
        pipeline = VisualizationPipeline(simulation_results, config, arena)

        # Determine if wraparound handling is needed for visualization
        wraparound = args.arena in ["pbc", "pbc-line"]

        # Plot trajectories for the first replicate
        pipeline.plot_trajectories_with_odor(
            sim_index=0,
            show=False,
            save_path=os.path.join(plots_path, "trajectories.png"),
            wraparound=wraparound,
        )

        # Plot final odor grid
        pipeline.plot_final_odor_grid(
            sim_index=0,
            show=False,
            save_path=os.path.join(plots_path, "final_odor_grid.png"),
        )

        # Plot odor time series
        pipeline.plot_odor_time_series(
            sim_index=0,
            show=False,
            save_path=os.path.join(plots_path, "odor_time_series.png"),
        )

        logger.info(f"Visualizations saved to {plots_path}")


if __name__ == "__main__":
    main()
