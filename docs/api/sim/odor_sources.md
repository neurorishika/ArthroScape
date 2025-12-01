# Odor Sources

External odor source loading and manipulation for ArthroScape simulations.

This module provides classes and utilities for loading odor distributions from
external sources such as images and videos. These can be used to
create complex, realistic odor landscapes that would be difficult to generate
programmatically.

## Base Class

::: arthroscape.sim.odor_sources.OdorSource
    options:
      show_root_heading: true
      show_source: true
      members:
        - apply_to_arena
        - get_odor_map

## Image Source

::: arthroscape.sim.odor_sources.ImageOdorSource
    options:
      show_root_heading: true
      show_source: true

## Video Source

::: arthroscape.sim.odor_sources.VideoOdorSource
    options:
      show_root_heading: true
      show_source: true

## Video Release Strategy

::: arthroscape.sim.odor_sources.VideoOdorReleaseStrategy
    options:
      show_root_heading: true
      show_source: true

## Convenience Functions

::: arthroscape.sim.odor_sources.load_odor_from_image
    options:
      show_root_heading: true
      show_source: true

::: arthroscape.sim.odor_sources.create_gradient_odor_map
    options:
      show_root_heading: true
      show_source: true
