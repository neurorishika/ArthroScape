# arthroscape/sim/main.py
import argparse
import os
import logging
from arthroscape.sim.config import SimulationConfig
from arthroscape.sim.arena import create_circular_arena_with_annular_trail
from arthroscape.sim.behavior import DefaultBehavior
from arthroscape.sim.odor_release import DefaultOdorRelease, ConstantOdorRelease
from arthroscape.sim.runner import run_simulations, save_simulation_results
from arthroscape.sim.visualization import VisualizationPipeline  # or visualizer if using that module

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Multi-Animal Fly Simulation in a Circular Arena")
    parser.add_argument("--replicates", type=int, default=1, help="Number of simulation replicates")
    parser.add_argument("--animals", type=int, default=1, help="Number of animals per replicate")
    parser.add_argument("--parallel", action="store_true", help="Run simulations in parallel")
    parser.add_argument("--odor_release", type=str, default="constant", choices=["none", "conditional", "constant"],
                        help="Odor release strategy: 'none', 'conditional', or 'constant'")
    parser.add_argument("--deposit_amount", type=float, default=0.5,
                        help="Deposit amount for constant odor release (pheromone)")
    parser.add_argument("--save", type=str, default="", help="Filename to save results (e.g., results.npz)")
    parser.add_argument("--visualize", action="store_true", help="Visualize simulation results")
    args = parser.parse_args()

    # Update number of animals in the configuration.
    config = SimulationConfig(number_of_animals=args.animals)

    # Get behavioral algorithm.
    behavior = DefaultBehavior()

    # Create circular arena with an annular trail.
    arena = create_circular_arena_with_annular_trail(config,
                                                     arena_radius=75.0,
                                                     trail_radius=42.5,
                                                     trail_width=5.0,
                                                     trail_odor=1.0)
    # Select odor release strategy.
    if args.odor_release == "none":
        odor_release_strategy = DefaultOdorRelease()
    elif args.odor_release == "constant":
        odor_release_strategy = ConstantOdorRelease(deposit_amount=args.deposit_amount)
    else:
        raise ValueError("Unknown odor release strategy selected.")

    logging.info("Starting simulation...")
    simulation_results = run_simulations(config, behavior, arena, odor_release_strategy,
                                         n_replicates=args.replicates, parallel=args.parallel)
    logging.info("Simulation complete.")

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        save_simulation_results(simulation_results, args.save)

    if args.visualize:
        viz = VisualizationPipeline(sim_results=simulation_results, config=config, arena=arena)
        viz.plot_trajectories_with_odor(sim_index=0, show=True)
        viz.plot_final_odor_grid(show=True)
        viz.plot_odor_time_series(sim_index=0, show=True)
        viz.animate_enhanced_trajectory(sim_index=0, interval=1, frame_skip=30, save_path="trajectory_animation.gif")

if __name__ == "__main__":
    main()
