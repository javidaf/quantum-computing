import numpy as np
from qiskit import QuantumCircuit
from abc import ABC, abstractmethod


class BaseAnsatz(ABC):
    """Base class for quantum ansatzes (parameterized quantum circuits)"""

    @abstractmethod
    def apply(self, circuit: QuantumCircuit, parameters: np.ndarray) -> QuantumCircuit:
        """Apply parameterized quantum gates to the circuit"""
        pass

    @abstractmethod
    def num_parameters(self, num_qubits: int) -> int:
        """Return the number of parameters needed for this ansatz"""
        pass


class SimpleAnsatz(BaseAnsatz):
    """
    Simple ansatz with rotation gates and CNOT gates.
    Structure: RY gates -> CNOT gates -> RY gates -> CNOT gates
    """

    def __init__(self, num_layers=1):
        """
        Args:
            num_layers: Number of layers to repeat the ansatz
        """
        self.num_layers = num_layers

    def num_parameters(self, num_qubits: int) -> int:
        """Each layer needs num_qubits parameters (RY gates before and after CNOTs)"""
        return num_qubits * self.num_layers

    def apply(self, circuit: QuantumCircuit, parameters: np.ndarray) -> QuantumCircuit:
        """
        Apply the simple ansatz to the circuit.

        Args:
            circuit: Quantum circuit to modify
            parameters: Parameter array for the ansatz

        Returns:
            Modified quantum circuit
        """
        num_qubits = circuit.num_qubits
        param_idx = 0

        for layer in range(self.num_layers):
            for qubit in range(num_qubits):
                if param_idx < len(parameters):
                    circuit.ry(parameters[param_idx], qubit)
                    param_idx += 1

            for qubit in range(num_qubits - 1):
                circuit.cx(qubit, qubit + 1)

        return circuit


class AdvancedAnsatz(BaseAnsatz):
    """
    Structure for each layer:
    1. RY gates on all qubits.
    2. CNOT gates connecting all pairs of qubits (control < target).
    3. RY gates on all qubits.
    """

    def __init__(self, num_layers: int = 1):
        """
        Args:
            num_layers: Number of times to repeat the RY-CNOTs-RY block.
        """
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer.")
        self.num_layers = num_layers

    def num_parameters(self, num_qubits: int) -> int:
        """
        Calculates the total number of parameters.
        Each layer has two sets of RY gates, each using num_qubits parameters.
        So, 2 * num_qubits parameters per layer.
        """
        if not isinstance(num_qubits, int) or num_qubits < 1:
            raise ValueError("num_qubits must be a positive integer.")
        return 2 * num_qubits * self.num_layers

    def apply(self, circuit: QuantumCircuit, parameters: np.ndarray) -> QuantumCircuit:
        """
        Apply the ansatz to the quantum circuit.

        Args:
            circuit: The QuantumCircuit to apply the ansatz to.
            parameters: A 1D numpy array of parameters for the RY gates.
                        The length should be num_qubits * 2 * num_layers.

        Returns:
            The modified QuantumCircuit.

        Raises:
            ValueError: If the number of parameters is incorrect.
        """
        num_qubits = circuit.num_qubits
        expected_params = self.num_parameters(num_qubits)

        if not isinstance(parameters, np.ndarray):
            raise TypeError("Parameters must be a numpy array.")
        if len(parameters) != expected_params:
            raise ValueError(
                f"Incorrect number of parameters. Expected {expected_params}, got {len(parameters)}."
            )

        param_idx = 0
        for _ in range(self.num_layers):

            for qubit in range(num_qubits):
                circuit.ry(parameters[param_idx], qubit)
                param_idx += 1

            # CNOT gates for entanglement (all-to-all, control_qubit < target_qubit)
            if num_qubits > 1:
                for control_qubit in range(num_qubits):
                    for target_qubit in range(control_qubit + 1, num_qubits):
                        circuit.cx(control_qubit, target_qubit)

            # Second layer of RY gates
            for qubit in range(num_qubits):
                circuit.ry(parameters[param_idx], qubit)
                param_idx += 1

        return circuit
