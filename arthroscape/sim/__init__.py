# arthroscape/sim/__init__.py
"""
ArthroScape Simulation Package.

This package provides the core simulation functionality for modeling
arthropod behavior in odor-rich environments.

Modules:
    config: Simulation configuration (SimulationConfig).
    arena: Arena environments (GridArena, PeriodicSquareArena, etc.).
    behavior: Behavioral algorithms (BehaviorAlgorithm, DefaultBehavior).
    simulator: Core simulation engine (MultiAnimalSimulator).
    runner: High-level simulation runners.
    odor_release: Odor release strategies.
    odor_perception: Odor perception models.
    odor_sources: External odor sources (images, videos).
    visualization: Plotting and animation tools.
    saver: Result saving utilities.
"""

from .config import SimulationConfig
from .arena import Arena, GridArena, PeriodicSquareArena
from .behavior import BehaviorAlgorithm, DefaultBehavior
from .simulator import MultiAnimalSimulator
from .odor_release import OdorReleaseStrategy, DefaultOdorRelease, ConstantOdorRelease
from .odor_sources import (
    OdorSource,
    ImageOdorSource,
    VideoOdorSource,
    VideoOdorReleaseStrategy,
    load_odor_from_image,
    create_gradient_odor_map,
)

__all__ = [
    # Config
    "SimulationConfig",
    # Arena
    "Arena",
    "GridArena",
    "PeriodicSquareArena",
    # Behavior
    "BehaviorAlgorithm",
    "DefaultBehavior",
    # Simulator
    "MultiAnimalSimulator",
    # Odor Release
    "OdorReleaseStrategy",
    "DefaultOdorRelease",
    "ConstantOdorRelease",
    # Odor Sources
    "OdorSource",
    "ImageOdorSource",
    "VideoOdorSource",
    "VideoOdorReleaseStrategy",
    "load_odor_from_image",
    "create_gradient_odor_map",
]
