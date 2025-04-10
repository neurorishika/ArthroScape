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

class AblatedPerception(AgentOdorPerception):
    def __init__(self, direction: str = "random") -> None:
        self.direction = direction
        if self.direction not in ["left", "right", "random"]:
            raise ValueError("Direction must be 'left', 'right', or 'random'.")
        if self.direction == "random":
            self.direction = np.random.choice(["left", "right"])

    def reset(self) -> None:
        pass

    def perceive_odor(self, raw_left: float, raw_right: float, dt: float) -> float:
        # Return zero for the ablated side
        if self.direction == "left":
            return (0.0, raw_right)
        elif self.direction == "right":
            return (raw_left, 0.0)
        else:
            raise ValueError("Invalid direction for ablation.")

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
    def __init__(self, tau_adapt: float = 0.5, tau_recovery: float = 2.0, beta: float = 1.0):
        """
        :param tau_adapt: Time constant (in seconds) for adaptation build-up (fast dynamics).
        :param tau_recovery: Time constant (in seconds) for recovery (slow dynamics).
        :param beta: Scaling factor that controls how strongly the adaptation variable reduces the response.
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

    def perceive_odor(self, raw_left: float, raw_right: float, dt: float) -> tuple:
        """
        Update the adaptation state and compute the perceived odor.
        
        The adaptation variable for each sensor is updated with two processes:
         - A fast adaptation term that causes the variable to move toward the raw stimulus.
         - A slower recovery term that causes the adaptation variable to decay back toward 0.
         
        The perceived odor is then computed as the raw odor divided by a factor (1 + beta * adaptation).
        This results in reduced perceived odor when adaptation is high.
        """
        # Update adaptation state for left sensor:
        dA_left = (raw_left - self.adapt_left) / self.tau_adapt  # fast build-up toward current stimulus
        recovery_left = -self.adapt_left / self.tau_recovery       # slow recovery (decay)
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
