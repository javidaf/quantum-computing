import numpy as np
from typing import Optional


class Optimizer:
    """
    Optimizer that uses simple gradient descent.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
    ):
        """
        Initialize the Optimizer.

        Args:
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.num_params = 0

    def initialize(self, initial_params: np.ndarray):
        """
        Initialize optimizer state.

        Args:
            initial_params: Initial parameters of the model.
        """
        self.num_params = len(initial_params)

    def update(self, current_params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Update parameters using gradient descent.

        Args:
            current_params: Current parameters.
            gradient: Gradient of the loss function.

        Returns:
            Updated parameters.
        """
        if len(current_params) != self.num_params or len(gradient) != self.num_params:
            raise ValueError(
                "Parameter or gradient dimension mismatch with initialized num_params."
            )
        return current_params - self.learning_rate * gradient


class AdamOptimizer(Optimizer):
    """
    Advanced Optimizer using Adam optimizer.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """
        Initialize Adam Optimizer.

        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)  # Pass learning_rate to parent
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Adam state will be initialized in the 'initialize' method
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t: int = 0

    def initialize(self, initial_params: np.ndarray):
        """
        Initialize Adam optimizer state.

        Args:
            initial_params: Initial parameters of the model.
        """
        super().initialize(initial_params)
        self.m = np.zeros(self.num_params)  # First moment vector
        self.v = np.zeros(self.num_params)  # Second moment vector
        self.t = 0  # Time step

    def update(self, current_params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Update parameters using Adam optimizer.

        Args:
            current_params: Current parameters.
            gradient: Gradient of the loss function.

        Returns:
            Updated parameters.
        """
        if self.m is None or self.v is None:
            raise RuntimeError("Optimizer not initialized. Call 'initialize' first.")
        if len(current_params) != self.num_params or len(gradient) != self.num_params:
            raise ValueError(
                "Parameter or gradient dimension mismatch with initialized num_params."
            )

        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)

        # Compute bias-corrected first moment estimate
        m_corrected = self.m / (1 - self.beta1**self.t)

        # Compute bias-corrected second raw moment estimate
        v_corrected = self.v / (1 - self.beta2**self.t)

        # Update parameters
        new_params = current_params - self.learning_rate * m_corrected / (
            np.sqrt(v_corrected) + self.epsilon
        )
        return new_params
