# tests/test_simulation.py
import os
import pickle
import tempfile
import shutil
import functools
import numpy as np
import h5py
import pytest
import math
from math import pi, isclose

# Import modules from the package
from arthroscape.sim.config import (
    SimulationConfig,
    get_walking_speed_sampler,
    get_turn_angle_sampler,
    get_initial_position_sampler,
    get_initial_heading_sampler
)
from arthroscape.sim.arena import (
    GridArena,
    create_circular_arena_with_annular_trail,
    PeriodicSquareArena,
    wrap_coordinates
)
from arthroscape.sim.directional_persistence import (
    FixedBlendPersistence,
    OdorDifferenceWeightedPersistence,
    AvgOdorWeightedPersistence
)
from arthroscape.sim.odor_perception import (
    NoAdaptationPerception,
    LowPassPerception,
    DerivativePerception,
    AdaptationPerception
)
from arthroscape.sim.odor_release import (
    ConstantOdorRelease,
    DefaultOdorRelease,
    _get_normalized_gaussian_kernel
)
from arthroscape.sim.behavior import DefaultBehavior
from arthroscape.sim.simulator import MultiAnimalSimulator
from arthroscape.sim.runner import run_simulations
from arthroscape.sim.saver import save_simulation_results_hdf5
from arthroscape.sim.visualization import (
    VisualizationPipeline,
    segment_trajectory_with_indices
)

###########################################
# CONFIGURATION TESTS
###########################################
def test_config_defaults():
    """Ensure default values and derived parameters are computed correctly."""
    config = SimulationConfig()
    config.__post_init__()
    # Kernel size should be odd.
    assert config.deposit_kernel_size % 2 == 1
    # Total frames should equal T * fps.
    expected_frames = int(config.T * config.fps)
    assert config.total_frames == expected_frames
    # With infinite odor_decay_tau, odor_decay_rate must be zero.
    assert config.odor_decay_rate == 0.0

def test_config_with_extreme_sigma():
    """Test that changing deposit_sigma changes the deposit_kernel_size accordingly."""
    config = SimulationConfig(deposit_sigma=0.2)
    config.__post_init__()
    # Expect a smaller kernel than the default.
    default_config = SimulationConfig()
    default_config.__post_init__()
    assert config.deposit_kernel_size < default_config.deposit_kernel_size

def test_config_pickleable():
    """Ensure that a SimulationConfig object is pickleable."""
    config = SimulationConfig()
    config.__post_init__()
    try:
        pickle.dumps(config)
    except Exception as e:
        pytest.fail(f"SimulationConfig not pickleable: {e}")

def test_sampler_functions():
    """Test the helper sampler functions."""
    ws = get_walking_speed_sampler(15)()
    assert ws == 15

    ta = get_turn_angle_sampler(0.1, 0.5)()
    assert 0.1 <= ta <= 0.5

    ip = get_initial_position_sampler(-80, 80, -80, 80)()
    assert isinstance(ip, tuple) and len(ip) == 2

    ih = get_initial_heading_sampler()()
    assert -pi <= ih <= pi

###########################################
# ARENA TESTS
###########################################
def test_arena_deposit_and_interpolation():
    """Deposit a normalized kernel and verify grid update and interpolation."""
    config = SimulationConfig(T=1, fps=1)
    config.__post_init__()
    arena = GridArena(config.grid_x_min, config.grid_x_max,
                      config.grid_y_min, config.grid_y_max,
                      config.grid_resolution, config=config)
    # Create a 3x3 kernel of ones (normalized).
    kernel = np.ones((3, 3))
    kernel /= kernel.sum()
    cx = (config.grid_x_min + config.grid_x_max) / 2
    cy = (config.grid_y_min + config.grid_y_max) / 2
    arena.deposit_odor_kernel(cx, cy, kernel)
    np.testing.assert_allclose(arena.odor_grid.sum(), 1, rtol=1e-5)
    # Interpolation should give nonzero value.
    odor_val = arena.get_odor(cx, cy)
    assert odor_val > 0

def test_diffusion_update():
    """Check that diffusion and decay update the odor grid appropriately."""
    config = SimulationConfig(T=1, fps=1, diffusion_coefficient=0.1, odor_decay_tau=10)
    config.__post_init__()
    arena = GridArena(config.grid_x_min, config.grid_x_max,
                      config.grid_y_min, config.grid_y_max,
                      config.grid_resolution, config=config)
    cx = (config.grid_x_min + config.grid_x_max) / 2
    cy = (config.grid_y_min + config.grid_y_max) / 2
    arena.update_odor(cx, cy, 100)
    total_before = arena.odor_grid.sum()
    arena.update_odor_field(dt=1)
    total_after = arena.odor_grid.sum()
    assert total_after < total_before

def test_periodic_arena_wrapping():
    """Test that PeriodicSquareArena correctly wraps coordinates."""
    config = SimulationConfig(T=1, fps=1)
    config.__post_init__()
    arena = PeriodicSquareArena(config.grid_x_min, config.grid_x_max,
                                config.grid_y_min, config.grid_y_max,
                                config.grid_resolution, config=config)
    wrapped_x = wrap_coordinates(np.array([config.grid_x_max + 10]), config.grid_x_min, config.grid_x_max)
    wrapped_y = wrap_coordinates(np.array([config.grid_y_min - 10]), config.grid_y_min, config.grid_y_max)
    assert config.grid_x_min <= wrapped_x[0] < config.grid_x_max
    assert config.grid_y_min <= wrapped_y[0] < config.grid_y_max

###########################################
# DIRECTIONAL PERSISTENCE TESTS
###########################################
def test_fixed_blend_persistence():
    dp = FixedBlendPersistence(alpha=0.0)
    new_heading = dp.adjust_heading(pi/4, pi/2, 0, 0, None, np.random.default_rng())
    assert isclose(new_heading, pi/2, rel_tol=1e-5)
    dp = FixedBlendPersistence(alpha=1.0)
    new_heading = dp.adjust_heading(pi/4, pi/2, 0, 0, None, np.random.default_rng())
    assert isclose(new_heading, pi/4, rel_tol=1e-5)

def test_odor_difference_weighted_persistence():
    dp = OdorDifferenceWeightedPersistence(alpha_min=0.3, alpha_max=0.7)
    # When both sensors are equal, persistence should be high.
    h_equal = dp.adjust_heading(pi/4, pi/2, 5, 5, None, np.random.default_rng())
    # When there's a strong difference, persistence should be lower.
    h_diff = dp.adjust_heading(pi/4, pi/2, 10, 1, None, np.random.default_rng())
    # We expect h_diff to be closer to the computed heading (pi/2).
    assert abs(h_diff - (pi/2)) < abs(h_equal - (pi/2))

def test_avg_odor_weighted_persistence():
    dp = AvgOdorWeightedPersistence(alpha_min=0.3, alpha_max=0.7)
    h_low = dp.adjust_heading(pi/4, pi/2, 0.1, 0.1, None, np.random.default_rng())
    h_high = dp.adjust_heading(pi/4, pi/2, 10, 10, None, np.random.default_rng())
    # With high average odor, expect the heading to lean more toward prev_heading.
    assert h_high < h_low or isclose(h_high, h_low)

###########################################
# ODOR PERCEPTION TESTS
###########################################
def test_no_adaptation_perception():
    op = NoAdaptationPerception()
    op.reset()
    left, right = op.perceive_odor(5, 7, 1)
    assert left == 5 and right == 7

def test_low_pass_perception():
    op = LowPassPerception(alpha=0.5)
    op.reset()
    # First call; output should be halfway toward input.
    left, right = op.perceive_odor(10, 20, 1)
    np.testing.assert_allclose(left, 5, rtol=1e-2)
    np.testing.assert_allclose(right, 10, rtol=1e-2)

def test_derivative_perception():
    op = DerivativePerception(alpha=0.5)
    op.reset()
    op.perceive_odor(10, 20, 1)  # initialize
    left, right = op.perceive_odor(15, 30, 1)
    np.testing.assert_allclose(left, 5, rtol=1e-2)
    np.testing.assert_allclose(right, 10, rtol=1e-2)

def test_adaptation_perception():
    op = AdaptationPerception(tau_adapt=0.5, tau_recovery=2.0, beta=1.0)
    op.reset()
    left1, right1 = op.perceive_odor(10, 10, 1)
    left2, right2 = op.perceive_odor(10, 10, 1)
    # Over successive steps with the same high odor, perceived values should decrease.
    assert left2 < left1 and right2 < right1

###########################################
# ODOR RELEASE TESTS
###########################################
def test_constant_odor_release_kernel():
    config = SimulationConfig(T=1, fps=1)
    config.deposit_sigma = 0.5  # Small sigma
    config.__post_init__()
    odor_release = ConstantOdorRelease(config=config, deposit_amount=10)
    instructions = odor_release.release_odor(1, (0, 0), 0, config, np.random.default_rng())
    assert len(instructions) >= 1
    kernel = instructions[0].generate_kernel(config)
    # Kernel should have the shape determined by deposit_kernel_size.
    expected_shape = (config.deposit_kernel_size, config.deposit_kernel_size)
    assert kernel.shape == expected_shape
    # The sum of the kernel should equal the deposit amount (10).
    np.testing.assert_allclose(kernel.sum(), 10, rtol=1e-5)

def test_default_odor_release():
    odor_release = DefaultOdorRelease()
    instructions = odor_release.release_odor(0, (0,0), 0, SimulationConfig(), np.random.default_rng())
    assert instructions == []

###########################################
# BEHAVIOR TESTS
###########################################
def test_default_behavior_state():
    behavior = DefaultBehavior()
    config = SimulationConfig(T=1, fps=1)
    state = behavior.update_state(0, config, np.random.default_rng(42))
    assert state in (0, 1)

def test_default_behavior_heading():
    behavior = DefaultBehavior()
    config = SimulationConfig(T=1, fps=1)
    new_heading = behavior.update_heading(0, 5, 10, False, config, np.random.default_rng(42))
    assert isinstance(new_heading, float)

###########################################
# SIMULATOR & RUNNER TESTS
###########################################
def test_simulator_output_shape():
    config = SimulationConfig(T=2, fps=1, number_of_animals=1)
    config.__post_init__()
    arena = GridArena(config.grid_x_min, config.grid_x_max,
                      config.grid_y_min, config.grid_y_max,
                      config.grid_resolution, config=config)
    behavior = DefaultBehavior()
    odor_release = DefaultOdorRelease()
    simulator = MultiAnimalSimulator(config, behavior, arena, odor_release, seed=42)
    result = simulator.simulate()
    traj = result["trajectories"][0]
    assert len(traj["x"]) == config.total_frames
    assert result["final_odor_grid"].shape == (arena.ny, arena.nx)

def test_runner_parallel_serial():
    config = SimulationConfig(T=1, fps=1, number_of_animals=1)
    config.__post_init__()
    arena = GridArena(config.grid_x_min, config.grid_x_max,
                      config.grid_y_min, config.grid_y_max,
                      config.grid_resolution, config=config)
    behavior = DefaultBehavior()
    odor_release = DefaultOdorRelease()
    serial_results = run_simulations(config, behavior, arena, odor_release, n_replicates=2, parallel=False)
    parallel_results = run_simulations(config, behavior, arena, odor_release, n_replicates=2, parallel=True)
    assert len(serial_results) == 2
    assert len(parallel_results) == 2

###########################################
# SAVER TESTS
###########################################
def test_hdf5_saver(tmp_path):
    dummy_results = [{
        "final_odor_grid": np.zeros((10, 10)),
        "trajectories": [{
            "x": [0, 1, 2],
            "y": [0, 1, 2],
            "heading": [0, 0, 0],
            "state": [0, 1, 0],
            "odor_left": [1, 2, 3],
            "odor_right": [4, 5, 6],
            "perceived_odor_left": [1, 2, 3],
            "perceived_odor_right": [4, 5, 6]
        }]
    }]
    filename = tmp_path / "dummy_results.h5"
    save_simulation_results_hdf5(dummy_results, str(filename))
    with h5py.File(str(filename), 'r') as hf:
        assert "replicate_0" in hf
        rep0 = hf["replicate_0"]
        assert "final_odor_grid" in rep0
        assert "trajectories" in rep0
        assert "animal_0" in rep0["trajectories"]

###########################################
# VISUALIZATION TESTS
###########################################
def test_wrap_coordinates_function():
    arr = np.array([90, 95, 105, 110])
    wrapped = wrap_coordinates(arr, 100, 120)
    # Values below 100 should wrap close to the max.
    assert wrapped[0] >= 100 and wrapped[0] < 120
    np.testing.assert_allclose(wrapped[2:], arr[2:])

def test_segment_trajectory():
    x = np.array([0, 1, 2, 100, 101, 102])
    y = np.array([0, 1, 2, 0, 1, 2])
    segments = segment_trajectory_with_indices(x, y, 0, 110, 0, 10)
    assert len(segments) == 2
    assert segments[0] == (0, 3)
    assert segments[1] == (3, 6)

def test_visualization_plots(tmp_path):
    """Ensure visualization functions run without error and produce files."""
    dummy_results = [{
        "final_odor_grid": np.random.random((50, 50)),
        "trajectories": [{
            "x": list(np.linspace(0, 10, 50)),
            "y": list(np.linspace(0, 10, 50)),
            "heading": [0]*50,
            "state": [1]*50,
            "odor_left": list(np.random.random(50)),
            "odor_right": list(np.random.random(50))
        }]
    }]
    config = SimulationConfig(T=1, fps=1)
    config.__post_init__()
    arena = GridArena(config.grid_x_min, config.grid_x_max,
                      config.grid_y_min, config.grid_y_max,
                      config.grid_resolution, config=config)
    viz = VisualizationPipeline(dummy_results, config, arena)
    out_dir = tmp_path / "plots"
    out_dir.mkdir()
    viz.plot_trajectories_with_odor(sim_index=0, show=False, save_path=str(out_dir / "traj.png"), wraparound=False)
    viz.plot_final_odor_grid(sim_index=0, show=False, save_path=str(out_dir / "odor.png"))
    viz.plot_odor_time_series(sim_index=0, show=False, save_path=str(out_dir / "odor_ts.png"))
    # Check that the files were created.
    assert (out_dir / "traj.png").exists()
    assert (out_dir / "odor.png").exists()
    assert (out_dir / "odor_ts.png").exists()

###########################################
# END OF TEST SUITE
###########################################
if __name__ == "__main__":
    pytest.main([__file__])
