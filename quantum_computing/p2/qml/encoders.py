import numpy as np
from qiskit import QuantumCircuit
from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    """Base class for quantum data encoders"""

    @abstractmethod
    def encode(self, circuit: QuantumCircuit, features: np.ndarray) -> QuantumCircuit:
        """Encode classical features into quantum state"""
        pass


class AngleEncoder(BaseEncoder):
    """
    Encodes features using rotation gates after Hadamard gates.
    This creates a superposition state and then rotates based on feature values.
    """

    def __init__(self, rotation_gate="ry", scaling=2 * np.pi):
        """
        Args:
            rotation_gate: Type of rotation gate ('rz', 'ry', 'rx')
            scaling: Scaling factor for feature values (default: 2π)
        """
        self.rotation_gate = rotation_gate
        self.scaling = scaling

    def encode(self, circuit: QuantumCircuit, features: np.ndarray) -> QuantumCircuit:
        """
        Encode features into quantum state using angle encoding.

        Args:
            circuit: Quantum circuit to modify
            features: Feature vector to encode

        Returns:
            Modified quantum circuit
        """
        # print(f"Encoding features: {features}")
        num_features = len(features)

        for qubit_idx in range(min(num_features, circuit.num_qubits)):
            circuit.h(qubit_idx)

        # Apply rotation gates based on feature values
        for feature_idx, feature_value in enumerate(features):

            if feature_idx >= circuit.num_qubits:
                break

            angle = float(self.scaling * feature_value)

            if self.rotation_gate == "rz":
                circuit.rz(angle, feature_idx)
            elif self.rotation_gate == "ry":
                circuit.ry(angle, feature_idx)
            elif self.rotation_gate == "rx":
                circuit.rx(angle, feature_idx)
            else:
                raise ValueError(f"Unsupported rotation gate: {self.rotation_gate}")

        return circuit


class HadRotZRotZZEncoder(BaseEncoder):
    """
    Implements the feature map based on the provided image.
    The encoding consists of three main parts:
    1. Hadamard gates applied to all qubits that will be encoded.
    2. RZ gates on each of these qubits, with rotation angles proportional
       to the individual feature values (x_i).
    3. A series of entangling blocks. For each pair of qubits (i, j) where i < j,
       an entangling block CNOT(i,j) - RZ(angle_ij) - CNOT(i,j) is applied.
       The RZ gate is on qubit j, and its angle is proportional to the
       product of the corresponding features (x_i * x_j).
    """

    def __init__(self, scaling: float = 2 * np.pi):
        """
        Args:
            scaling: Scaling factor for the feature values. This factor multiplies
                     both the individual features (for single-qubit RZ gates)
                     and the product of features (for two-qubit interaction RZ gates).
                     Default is 2π.
        """
        self.scaling = scaling

    def encode(self, circuit: QuantumCircuit, features: np.ndarray) -> QuantumCircuit:
        """
        Encodes the given classical features into the quantum state of the circuit.

        Args:
            circuit: The QuantumCircuit object to apply the encoding to.
            features: A 1D numpy array of classical features to encode.
                      The length of this array can be less than, equal to,
                      or greater than the number of qubits in the circuit.
                      Encoding will be applied to min(len(features), num_qubits) qubits.

        Returns:
            The modified QuantumCircuit with the encoding operations applied.
        """
        num_active_qubits = min(len(features), circuit.num_qubits)

        # Apply Hadamard gates to all active qubits
        for i in range(num_active_qubits):
            circuit.h(i)
        # Apply RZ gates based on feature values to each active qubit and scale them
        for i in range(num_active_qubits):
            angle = self.scaling * features[i]
            circuit.rz(angle, i)

        if num_active_qubits >= 2:
            for i in range(num_active_qubits):
                for j in range(i + 1, num_active_qubits):

                    interaction_angle = self.scaling * features[i] * features[j]

                    # RZZ interaction block
                    circuit.cx(i, j)
                    circuit.rz(interaction_angle, j)
                    circuit.cx(i, j)

        return circuit


class NoEncoder(BaseEncoder):
    """No encoding, returns the circuit unchanged."""

    def encode(self, circuit: QuantumCircuit, features: np.ndarray) -> QuantumCircuit:
        """
        No encoding applied, returns the original circuit.

        Args:
            circuit: Quantum circuit to modify
            features: Feature vector (not used)

        Returns:
            Original quantum circuit
        """
        return circuit
