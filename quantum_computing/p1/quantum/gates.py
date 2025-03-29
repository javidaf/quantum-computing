import numpy as np


class QuantumGate:
    """Base class for quantum gates"""

    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=complex)

    def apply(self, state):
        """Apply the gate to a quantum state"""
        return np.dot(self.matrix, state)

    def __str__(self):
        return f"{self.__class__.__name__} Gate"

    def __repr__(self):
        return self.__str__()


class PauliX(QuantumGate):
    """Pauli X gate (NOT gate)"""

    def __init__(self):
        super().__init__([[0, 1], [1, 0]])


class PauliY(QuantumGate):
    """Pauli Y gate"""

    def __init__(self):
        super().__init__([[0, -1j], [1j, 0]])


class PauliZ(QuantumGate):
    """Pauli Z gate"""

    def __init__(self):
        super().__init__([[1, 0], [0, -1]])


class Hadamard(QuantumGate):
    """Hadamard gate"""

    def __init__(self):
        super().__init__([[1, 1], [1, -1]] / np.sqrt(2))


class Phase(QuantumGate):
    """Phase gate (S gate)"""

    def __init__(self):
        super().__init__([[1, 0], [0, 1j]])


class CNOT(QuantumGate):
    """Controlled-NOT gate"""

    def __init__(self):
        super().__init__([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


class RX(QuantumGate):
    """RX rotation gate"""

    def __init__(self, theta):
        super().__init__(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )


class RY(QuantumGate):
    """RY rotation gate"""

    def __init__(self, theta):
        super().__init__(
            [
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )


class RZ(QuantumGate):
    """RZ rotation gate"""

    def __init__(self, theta):
        super().__init__([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
