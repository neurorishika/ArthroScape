# arthroscape/sim/odor_perception.py
from abc import ABC, abstractmethod
import numpy as np

class AgentOdorPerception(ABC):
    """
    Abstract base class for each agent's odor perception system.
    That is, each agent has its own instance of this class.
    """
    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state at the start of a simulation."""
        pass

    @abstractmethod
    def perceive_odor(self, raw_left: float, raw_right: float, dt: float) -> float:
        """
        :param raw_left:  The raw odor intensity for the left sensor.
        :param raw_right: The raw odor intensity for the right sensor.
        :param dt:        The time step (in seconds).
        :return: The final “perceived” odor value for this agent.
        """
        pass


class NoAdaptationPerception(AgentOdorPerception):
    def reset(self) -> None:
        pass

    def perceive_odor(self, raw_left: float, raw_right: float, dt: float) -> float:
        # Just return the original values
        return (raw_left, raw_right)


class LowPassPerception(AgentOdorPerception):
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.state_left = 0.0
        self.state_right = 0.0

    def reset(self) -> None:
        # Reset the internal state variables to zero.
        self.state_left = 0.0
        self.state_right = 0.0

    def perceive_odor(self, raw_left: float, raw_right: float, dt: float) -> float:
        # Apply low-pass filtering to the raw values
        self.state_left += self.alpha * (raw_left - self.state_left)
        self.state_right += self.alpha * (raw_right - self.state_right)
        return (self.state_left, self.state_right)


class DerivativePerception(AgentOdorPerception):
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.previous_left = 0.0
        self.previous_right = 0.0

    def reset(self) -> None:
        # Reset the internal state variables to zero.
        self.previous_left = 0.0
        self.previous_right = 0.0

    def perceive_odor(self, raw_left: float, raw_right: float, dt: float) -> float:
        # Compute the derivative of the raw values
        derivative_left = (raw_left - self.previous_left) / dt
        derivative_right = (raw_right - self.previous_right) / dt

        # Update the previous values for the next step
        self.previous_left = raw_left
        self.previous_right = raw_right

        return (derivative_left, derivative_right)

class AdaptationPerception(AgentOdorPerception):
    def __init__(self, alpha: float = 0.1, tau: float = 1.0):
        self.alpha = alpha 
        self.tau = tau
        # start with the gate fully open
        self.gate_left = 1.0
        self.gate_right = 1.0 

    def reset(self) -> None:
        # Reset the gate to fully open
        self.gate_left = 1.0
        self.gate_right = 0.0

    def perceive_odor(self, raw_left: float, raw_right: float, dt: float) -> float:
        # this is a neural model of adaptation
        # the longer the odor is present, the more the gate closes
        # the gate is a value between 0 and 1 that multiplies the raw value

        # update the gate values
        self.gate_left += dt/self.tau * ((1 - self.gate_left) - self.alpha * self.gate_left * raw_left)
        self.gate_right += dt/self.tau * ((1 - self.gate_right) - self.alpha * self.gate_right * raw_right)

        # apply the gate to the raw values
        return (raw_left * self.gate_left, raw_right * self.gate_right)
