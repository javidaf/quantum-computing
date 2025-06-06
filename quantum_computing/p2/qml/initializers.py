from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray


class BaseInitializer(ABC):
    """Base class for quantum circuit parameter initializers"""

    @abstractmethod
    def initialize(self, num_params: int) -> ndarray:
        """
        Initialize parameters for a quantum circuit.

        Args:
            num_params: Number of parameters in the circuit

        Returns:
            Array of initial parameters
        """
        pass


class RandomInitializer(BaseInitializer):
    def initialize(self, num_params: int) -> ndarray:
        """
        Initialize parameters randomly between 0 and 2π.

        Args:
            num_params: Number of parameters in the circuit

        Returns:
            Array of initial parameters
        """
        return np.random.uniform(0, 2 * np.pi, num_params)


class ZeroInitializer(BaseInitializer):
    def initialize(self, num_params: int) -> ndarray:
        """Initialize all parameters to zero."""
        return np.zeros(num_params)


class NormalInitializer(BaseInitializer):
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std

    def initialize(self, num_params: int) -> ndarray:
        """Initialize parameters from normal distribution."""
        return np.random.normal(self.mean, self.std, num_params)


class XavierInitializer(BaseInitializer):
    def initialize(self, num_params: int) -> ndarray:
        """Initialize using Xavier/Glorot initialization."""
        limit = np.sqrt(6.0 / num_params)
        return np.random.uniform(-limit, limit, num_params)
