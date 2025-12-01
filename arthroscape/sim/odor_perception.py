# arthroscape/sim/odor_perception.py
"""
Odor perception module for the ArthroScape simulation.

This module defines the `AgentOdorPerception` abstract base class and various concrete implementations.
These classes model how an agent processes raw odor signals from its sensors (antennae).
Implementations include simple pass-through, ablation (blocking one sensor), low-pass filtering,
derivative sensing, and adaptation models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class AgentOdorPerception(ABC):
    """
    Abstract base class for an agent's odor perception system.

    Each agent in the simulation holds its own instance of a perception class, allowing for
    individual state (e.g., for adaptation or filtering).
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal state of the perception system.

        This should be called at the start of a new simulation run or trial.
        """
        pass

    @abstractmethod
    def perceive_odor(
        self, raw_left: float, raw_right: float, dt: float
    ) -> Tuple[float, float]:
        """
        Process raw odor inputs and return perceived values.

        Args:
            raw_left (float): The raw odor concentration at the left sensor.
            raw_right (float): The raw odor concentration at the right sensor.
            dt (float): The time step duration in seconds.

        Returns:
            Tuple[float, float]: A tuple (perceived_left, perceived_right) representing the
                                 processed odor signals used for behavior.
        """
        pass


class NoAdaptationPerception(AgentOdorPerception):
    """
    A simple perception model that passes raw odor values through without modification.
    """

    def reset(self) -> None:
        pass

    def perceive_odor(
        self, raw_left: float, raw_right: float, dt: float
    ) -> Tuple[float, float]:
        """
        Return raw odor values directly.

        Args:
            raw_left (float): Raw left odor.
            raw_right (float): Raw right odor.
            dt (float): Time step.

        Returns:
            Tuple[float, float]: (raw_left, raw_right).
        """
        return (raw_left, raw_right)


class AblatedPerception(AgentOdorPerception):
    """
    A perception model that simulates sensory ablation by zeroing out the signal from one side.
    """

    def __init__(self, direction: str = "random") -> None:
        """
        Initialize the ablated perception model.

        Args:
            direction (str): The side to ablate. Options are "left", "right", or "random".
                             If "random", the side is chosen at initialization.

        Raises:
            ValueError: If direction is not 'left', 'right', or 'random'.
        """
        self.direction = direction
        if self.direction not in ["left", "right", "random"]:
            raise ValueError("Direction must be 'left', 'right', or 'random'.")
        if self.direction == "random":
            self.direction = np.random.choice(["left", "right"])

    def reset(self) -> None:
        pass

    def perceive_odor(
        self, raw_left: float, raw_right: float, dt: float
    ) -> Tuple[float, float]:
        """
        Return odor values with one side set to zero.

        Args:
            raw_left (float): Raw left odor.
            raw_right (float): Raw right odor.
            dt (float): Time step.

        Returns:
            Tuple[float, float]: (0.0, raw_right) if left is ablated, or (raw_left, 0.0) if right is ablated.
        """
        # Return zero for the ablated side
        if self.direction == "left":
            return (0.0, raw_right)
        elif self.direction == "right":
            return (raw_left, 0.0)
        else:
            raise ValueError("Invalid direction for ablation.")


class LowPassPerception(AgentOdorPerception):
    """
    A perception model that applies a first-order low-pass filter to the odor signals.
    This simulates the temporal integration properties of sensory neurons.
    """

    def __init__(self, tau: float = 0.1):
        """
        Initialize the low-pass filter.

        Args:
            tau (float): The time constant of the filter in seconds.
        """
        self.tau = tau
        self.state_left = 0.0
        self.state_right = 0.0

    def reset(self) -> None:
        """Reset filter states to zero."""
        self.state_left = 0.0
        self.state_right = 0.0

    def perceive_odor(
        self, raw_left: float, raw_right: float, dt: float
    ) -> Tuple[float, float]:
        """
        Apply low-pass filtering.

        Args:
            raw_left (float): Raw left odor.
            raw_right (float): Raw right odor.
            dt (float): Time step.

        Returns:
            Tuple[float, float]: Filtered (left, right) odor values.
        """
        # Apply low-pass filtering to the raw values
        self.state_left += dt / self.tau * (raw_left - self.state_left)
        self.state_right += dt / self.tau * (raw_right - self.state_right)
        return (self.state_left, self.state_right)


class DerivativePerception(AgentOdorPerception):
    """
    A perception model that computes the temporal derivative of the odor signal.
    """

    def __init__(self, scale: float = 0.1):
        """
        Initialize the derivative perception model.

        Args:
            scale (float): Scaling factor for the derivative.
        """
        self.previous_left = 0.0
        self.previous_right = 0.0
        self.scale = scale

    def reset(self) -> None:
        """Reset previous values to zero."""
        self.previous_left = 0.0
        self.previous_right = 0.0

    def perceive_odor(
        self, raw_left: float, raw_right: float, dt: float
    ) -> Tuple[float, float]:
        """
        Compute the derivative of the odor signal.

        Args:
            raw_left (float): Raw left odor.
            raw_right (float): Raw right odor.
            dt (float): Time step.

        Returns:
            Tuple[float, float]: (derivative_left, derivative_right).
        """
        # Compute the derivative of the raw values
        derivative_left = self.scale * (raw_left - self.previous_left) / dt
        derivative_right = self.scale * (raw_right - self.previous_right) / dt

        # Update the previous values for the next step
        self.previous_left = raw_left
        self.previous_right = raw_right

        return (derivative_left, derivative_right)


class AblatedDerivativePerception(AgentOdorPerception):
    """
    A perception model combining derivative sensing with unilateral ablation.
    """

    def __init__(self, scale: float = 0.1, direction: str = "random") -> None:
        """
        Initialize the ablated derivative perception model.

        Args:
            scale (float): Scaling factor for the derivative.
            direction (str): Side to ablate ('left', 'right', or 'random').
        """
        self.scale = scale
        self.direction = direction
        if self.direction not in ["left", "right", "random"]:
            raise ValueError("Direction must be 'left', 'right', or 'random'.")
        if self.direction == "random":
            self.direction = np.random.choice(["left", "right"])
        self.previous_left = 0.0
        self.previous_right = 0.0

    def reset(self) -> None:
        """Reset previous values to zero."""
        self.previous_left = 0.0
        self.previous_right = 0.0

    def perceive_odor(
        self, raw_left: float, raw_right: float, dt: float
    ) -> Tuple[float, float]:
        """
        Compute derivative and apply ablation.

        Args:
            raw_left (float): Raw left odor.
            raw_right (float): Raw right odor.
            dt (float): Time step.

        Returns:
            Tuple[float, float]: (derivative_left, derivative_right) with one side zeroed.
        """
        # Compute the derivative of the raw values
        derivative_left = self.scale * (raw_left - self.previous_left) / dt
        derivative_right = self.scale * (raw_right - self.previous_right) / dt

        # Update the previous values for the next step
        self.previous_left = raw_left
        self.previous_right = raw_right

        # Return zero for the ablated side
        if self.direction == "left":
            return (0.0, derivative_right)
        elif self.direction == "right":
            return (derivative_left, 0.0)
        else:
            raise ValueError("Invalid direction for ablation.")


class ScaleAdaptationPerception(AgentOdorPerception):
    """
    A perception model that implements gain control adaptation.

    The perceived intensity is scaled down based on an adaptation variable that tracks
    the recent history of the stimulus.
    """

    def __init__(
        self, tau_adapt: float = 0.5, tau_recovery: float = 2.0, beta: float = 1.0
    ):
        """
        Initialize the scale adaptation model.

        Args:
            tau_adapt (float): Time constant (in seconds) for adaptation build-up (fast dynamics).
            tau_recovery (float): Time constant (in seconds) for recovery (slow dynamics).
            beta (float): Scaling factor that controls how strongly the adaptation variable reduces the response.
        """
        self.tau_adapt = tau_adapt
        self.tau_recovery = tau_recovery
        self.beta = beta
        # Initialize adaptation variables for left and right sensors.
        self.adapt_left = 0.0
        self.adapt_right = 0.0

    def reset(self) -> None:
        """Reset the adaptation state (e.g., at the start of a simulation)."""
        self.adapt_left = 0.0
        self.adapt_right = 0.0

    def perceive_odor(
        self, raw_left: float, raw_right: float, dt: float
    ) -> Tuple[float, float]:
        """
        Update the adaptation state and compute the perceived odor.

        The adaptation variable for each sensor is updated with two processes:
         - A fast adaptation term that causes the variable to move toward the raw stimulus.
         - A slower recovery term that causes the adaptation variable to decay back toward 0.

        The perceived odor is then computed as the raw odor divided by a factor (1 + beta * adaptation).
        This results in reduced perceived odor when adaptation is high.

        Args:
            raw_left (float): Raw left odor.
            raw_right (float): Raw right odor.
            dt (float): Time step.

        Returns:
            Tuple[float, float]: (perceived_left, perceived_right).
        """
        # Update adaptation state for left sensor:
        dA_left = (
            raw_left - self.adapt_left
        ) / self.tau_adapt  # fast build-up toward current stimulus
        recovery_left = -self.adapt_left / self.tau_recovery  # slow recovery (decay)
        self.adapt_left += dt * (dA_left + recovery_left)
        # Ensure adaptation stays nonnegative.
        self.adapt_left = max(self.adapt_left, 0.0)

        # Update adaptation state for right sensor:
        dA_right = (raw_right - self.adapt_right) / self.tau_adapt
        recovery_right = -self.adapt_right / self.tau_recovery
        self.adapt_right += dt * (dA_right + recovery_right)
        self.adapt_right = max(self.adapt_right, 0.0)

        # Compute perceived odor. When adaptation is high, the effective gain decreases.
        perceived_left = raw_left / (1 + self.beta * self.adapt_left)
        perceived_right = raw_right / (1 + self.beta * self.adapt_right)
        return (perceived_left, perceived_right)


class LeakAdaptationPerception(AgentOdorPerception):
    """
    AdaptationPerception implements an odor integration and adaptation mechanism.

    The dynamics follow:

        dO/dt = (-O + raw - A) / odor_integration_tau
        dA/dt = (-A + (adaptation_magnitude * O)) / adaptation_tau

    where:
      • O is the integrated odor signal.
      • A is the adaptation variable.

    The perceived odor at each sensor is simply given by the current integrated signal O.
    """

    def __init__(
        self,
        odor_integration_tau: float = 1.0,
        adaptation_tau: float = 5.0,
        adaptation_magnitude: float = 0.5,
    ) -> None:
        """
        Initialize the leak adaptation model.

        Args:
            odor_integration_tau (float): Time constant for odor integration (in seconds).
            adaptation_tau (float): Time constant for adaptation recovery (in seconds).
            adaptation_magnitude (float): Strength of the odor’s influence on adaptation.

        Raises:
            ValueError: If time constants are not positive.
        """
        if odor_integration_tau <= 0:
            raise ValueError("odor_integration_tau must be positive.")
        if adaptation_tau <= 0:
            raise ValueError("adaptation_tau must be positive.")

        self.odor_integration_tau = odor_integration_tau
        self.adaptation_tau = adaptation_tau
        self.adaptation_magnitude = adaptation_magnitude

        # Initialize state variables for each sensor.
        self.integrated_left = 0.0
        self.integrated_right = 0.0
        self.adapt_left = 0.0
        self.adapt_right = 0.0

    def reset(self) -> None:
        """Reset the integrated odor and adaptation variables to initial values."""
        self.integrated_left = 0.0
        self.integrated_right = 0.0
        self.adapt_left = 0.0
        self.adapt_right = 0.0

    def perceive_odor(
        self, raw_left: float, raw_right: float, dt: float
    ) -> Tuple[float, float]:
        """
        Integrate the odor signal and update adaptation variables using Euler integration.

        Args:
            raw_left (float): Raw odor intensity at the left sensor.
            raw_right (float): Raw odor intensity at the right sensor.
            dt (float): Time step in seconds (must be positive).

        Returns:
            Tuple[float, float]: The perceived odor intensities for the left and right sensors.
        """
        if dt <= 0:
            raise ValueError("dt must be positive.")

        # Update the integrated odor signals using Euler's method.
        d_integrated_left = (
            -self.integrated_left + raw_left - self.adapt_left
        ) / self.odor_integration_tau
        d_integrated_right = (
            -self.integrated_right + raw_right - self.adapt_right
        ) / self.odor_integration_tau

        self.integrated_left += d_integrated_left * dt
        self.integrated_right += d_integrated_right * dt

        # Update adaptation variables.
        d_adapt_left = (
            -self.adapt_left + self.adaptation_magnitude * self.integrated_left
        ) / self.adaptation_tau
        d_adapt_right = (
            -self.adapt_right + self.adaptation_magnitude * self.integrated_right
        ) / self.adaptation_tau

        self.adapt_left += d_adapt_left * dt
        self.adapt_right += d_adapt_right * dt

        # Perceived odor is provided by the integrated (adapted) odor signal.
        return self.integrated_left, self.integrated_right


class AblatedLeakAdaptationPerception(AgentOdorPerception):
    """
    An ablated version of the LeakAdaptationPerception that zeros the integrated odor signal
    of one sensor. The ablated side is determined by the 'direction' parameter.
    """

    def __init__(
        self,
        odor_integration_tau: float = 1.0,
        adaptation_tau: float = 5.0,
        adaptation_magnitude: float = 0.5,
        direction: str = "random",
    ) -> None:
        """
        Initialize the ablated leak adaptation model.

        Args:
            odor_integration_tau (float): Time constant for odor integration (in seconds).
            adaptation_tau (float): Time constant for adaptation recovery (in seconds).
            adaptation_magnitude (float): Strength of the odor’s influence on adaptation.
            direction (str): The sensor to ablate: "left", "right", or "random".

        Raises:
            ValueError: If time constants are not positive or direction is invalid.
        """
        if odor_integration_tau <= 0:
            raise ValueError("odor_integration_tau must be positive.")
        if adaptation_tau <= 0:
            raise ValueError("adaptation_tau must be positive.")
        if direction not in ["left", "right", "random"]:
            raise ValueError("Direction must be 'left', 'right', or 'random'.")
        if direction == "random":
            direction = np.random.choice(["left", "right"])
        self.odor_integration_tau = odor_integration_tau
        self.adaptation_tau = adaptation_tau
        self.adaptation_magnitude = adaptation_magnitude
        self.direction = direction

        self.integrated_left = 0.0
        self.integrated_right = 0.0
        self.adapt_left = 0.0
        self.adapt_right = 0.0

    def reset(self) -> None:
        """Reset the integrated odor and adaptation variables to initial values."""
        self.integrated_left = 0.0
        self.integrated_right = 0.0
        self.adapt_left = 0.0
        self.adapt_right = 0.0

    def perceive_odor(
        self, raw_left: float, raw_right: float, dt: float
    ) -> Tuple[float, float]:
        """
        Integrate odor signal, update adaptation, and apply ablation.

        Args:
            raw_left (float): Raw left odor.
            raw_right (float): Raw right odor.
            dt (float): Time step.

        Returns:
            Tuple[float, float]: (perceived_left, perceived_right) with one side zeroed.
        """
        if dt <= 0:
            raise ValueError("dt must be positive.")

        d_integrated_left = (
            -self.integrated_left + raw_left - self.adapt_left
        ) / self.odor_integration_tau
        d_integrated_right = (
            -self.integrated_right + raw_right - self.adapt_right
        ) / self.odor_integration_tau

        self.integrated_left += d_integrated_left * dt
        self.integrated_right += d_integrated_right * dt

        d_adapt_left = (
            -self.adapt_left + self.adaptation_magnitude * self.integrated_left
        ) / self.adaptation_tau
        d_adapt_right = (
            -self.adapt_right + self.adaptation_magnitude * self.integrated_right
        ) / self.adaptation_tau

        self.adapt_left += d_adapt_left * dt
        self.adapt_right += d_adapt_right * dt

        if self.direction == "left":
            return (0.0, self.integrated_right)
        elif self.direction == "right":
            return (self.integrated_left, 0.0)
        else:
            # Should not happen if init checks pass
            raise ValueError("Invalid direction for ablation.")

        d_adapt_left = (
            -self.adapt_left + self.adaptation_magnitude * self.integrated_left
        ) / self.adaptation_tau
        d_adapt_right = (
            -self.adapt_right + self.adaptation_magnitude * self.integrated_right
        ) / self.adaptation_tau

        self.adapt_left += d_adapt_left * dt
        self.adapt_right += d_adapt_right * dt

        # Apply ablation by zeroing the integrated odor for the designated sensor.
        if self.direction == "left":
            perceived_left = 0.0
            perceived_right = self.integrated_right
        elif self.direction == "right":
            perceived_left = self.integrated_left
            perceived_right = 0.0
        else:
            raise ValueError("Invalid direction for ablation.")

        return perceived_left, perceived_right
