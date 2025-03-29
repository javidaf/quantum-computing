import numpy as np


class QuantumState:
    """Base class for quantum states"""

    def __init__(self, vector):
        self.vector = np.array(vector, dtype=complex)
        self._normalize()

    def _normalize(self):
        """Normalize the quantum state"""
        norm = np.sqrt(np.sum(np.abs(self.vector) ** 2))
        if norm > 0:
            self.vector = self.vector / norm

    @property
    def probabilities(self):
        """Return the probabilities of measuring each basis state"""
        return np.abs(self.vector) ** 2

    def __str__(self):
        return f"Quantum state: {self.vector}"

    def __repr__(self):
        return self.__str__()


class Qubit(QuantumState):
    """Single qubit state"""

    @classmethod
    def ket_0(cls):
        """Create a |0⟩ state"""
        return cls([1, 0])

    @classmethod
    def ket_1(cls):
        """Create a |1⟩ state"""
        return cls([0, 1])

    @classmethod
    def plus(cls):
        """Create a |+⟩ state: (|0⟩ + |1⟩)/√2"""
        return cls([1 / np.sqrt(2), 1 / np.sqrt(2)])

    @classmethod
    def minus(cls):
        """Create a |-⟩ state: (|0⟩ - |1⟩)/√2"""
        return cls([1 / np.sqrt(2), -1 / np.sqrt(2)])


class TwoQubits(QuantumState):
    """Two-qubit quantum state"""

    @classmethod
    def bell_state(cls, index=0):
        """Create one of the four Bell states
        index 0: (|00⟩ + |11⟩)/√2
        index 1: (|00⟩ - |11⟩)/√2
        index 2: (|01⟩ + |10⟩)/√2
        index 3: (|01⟩ - |10⟩)/√2
        """
        if index == 0:
            return cls([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
        elif index == 1:
            return cls([1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)])
        elif index == 2:
            return cls([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])
        elif index == 3:
            return cls([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0])
        else:
            raise ValueError("Bell state index must be between 0 and 3")

    @classmethod
    def ket_00(cls):
        return cls([1, 0, 0, 0])

    @classmethod
    def ket_01(cls):
        return cls([0, 1, 0, 0])

    @classmethod
    def ket_10(cls):
        return cls([0, 0, 1, 0])

    @classmethod
    def ket_11(cls):
        return cls([0, 0, 0, 1])
