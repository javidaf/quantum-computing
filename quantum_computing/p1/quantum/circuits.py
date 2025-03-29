import numpy as np

from quantum_computing.p1.quantum import gates
from quantum_computing.p1.quantum.states import QuantumState
from quantum_computing.p1.quantum.utils import tensor_product


class QuantumCircuit:
    """Class for constructing and simulating quantum circuits"""

    def __init__(self, num_qubits):
        """
        Initialize a quantum circuit with specified number of qubits

        Parameters:
        -----------
        num_qubits : int
            Number of qubits in the circuit
        """
        self.num_qubits = num_qubits
        self.gates = []
        self.initial_state = None

    def add_gate(self, gate, target_qubits, control_qubits=None):
        """
        Add a gate to the circuit

        Parameters:
        -----------
        gate : QuantumGate
            The quantum gate to add
        target_qubits : int or list
            Index or indices of target qubits
        control_qubits : int or list, optional
            Index or indices of control qubits
        """
        if not isinstance(target_qubits, list):
            target_qubits = [target_qubits]

        if control_qubits is not None and not isinstance(control_qubits, list):
            control_qubits = [control_qubits]

        self.gates.append((gate, target_qubits, control_qubits))

    def initialize(self, state):
        """
        Set the initial state of the circuit

        Parameters:
        -----------
        state : array-like or QuantumState
            The initial state vector or QuantumState object
        """
        if isinstance(state, QuantumState):
            self.initial_state = state.vector
        else:
            self.initial_state = np.array(state, dtype=complex)

    def _expand_gate(self, gate, target_qubits, control_qubits=None):
        """
        Expand a gate to act on the full Hilbert space

        Parameters:
        -----------
        gate : QuantumGate
            The quantum gate to expand
        target_qubits : list
            Indices of target qubits
        control_qubits : list, optional
            Indices of control qubits

        Returns:
        --------
        array-like : The expanded gate matrix
        """
        # Special case for CNOT gate
        if (
            isinstance(gate, gates.CNOT)
            and len(target_qubits) == 1
            and control_qubits
            and len(control_qubits) == 1
        ):
            # Create matrices for each possible CNOT configuration
            if self.num_qubits == 2:
                if control_qubits[0] == 0 and target_qubits[0] == 1:
                    # CNOT with control=first qubit, target=second qubit
                    return gate.matrix
                elif control_qubits[0] == 1 and target_qubits[0] == 0:
                    # CNOT with control=second qubit, target=first qubit
                    # Swap matrix representation
                    return np.array(
                        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
                        dtype=complex,
                    )
            elif self.num_qubits == 3:
                # For 3 qubits, we need different CNOT representations based on the control/target
                if control_qubits[0] == 0 and target_qubits[0] == 1:
                    # CNOT between qubit 0 (control) and 1 (target)
                    cnot_01 = np.kron(gate.matrix, np.eye(2, dtype=complex))
                    return cnot_01
                elif control_qubits[0] == 1 and target_qubits[0] == 2:
                    # CNOT between qubit 1 (control) and 2 (target)
                    cnot_12 = np.kron(np.eye(2, dtype=complex), gate.matrix)
                    return cnot_12
                elif control_qubits[0] == 2 and target_qubits[0] == 0:
                    # CNOT between qubit 2 (control) and 0 (target)
                    # This requires a more complex transformation
                    # First create a reordering of the basis states
                    perm = np.zeros((8, 8), dtype=complex)
                    for i in range(8):
                        # Extract bit values
                        bit0 = (i >> 0) & 1
                        bit1 = (i >> 1) & 1
                        bit2 = (i >> 2) & 1

                        # If bit2 is 1, flip bit0
                        if bit2:
                            bit0 = 1 - bit0

                        # Combine back to index
                        j = (bit2 << 2) | (bit1 << 1) | bit0
                        perm[j, i] = 1

                    return perm

        # Handle single-qubit gates
        if len(target_qubits) == 1 and target_qubits[0] < self.num_qubits:
            target = target_qubits[0]
            # Create a list of identity matrices except at the target
            matrices = []
            for i in range(self.num_qubits):
                if i == target:
                    matrices.append(gate.matrix)
                else:
                    matrices.append(np.eye(2, dtype=complex))

            # Build tensor product in the correct order
            if self.num_qubits == 2:
                return np.kron(matrices[0], matrices[1])
            elif self.num_qubits == 3:
                return np.kron(matrices[0], np.kron(matrices[1], matrices[2]))

        raise NotImplementedError(
            f"This gate configuration is not yet supported: {gate} on qubits {target_qubits} with controls {control_qubits}"
        )

    def simulate(self, initial_state=None):
        """
        Simulate the quantum circuit

        Parameters:
        -----------
        initial_state : array-like or QuantumState, optional
            Custom initial state (overrides the circuit's initial state)

        Returns:
        --------
        array-like : Final state vector after circuit execution
        """
        import numpy as np  # in case it's not imported already

        if initial_state is not None:
            if isinstance(initial_state, QuantumState):
                state = initial_state.vector
            else:
                state = np.array(initial_state, dtype=complex)
        elif self.initial_state is not None:
            state = self.initial_state
        else:
            # Default to |0...0⟩ state
            state = np.zeros(2**self.num_qubits, dtype=complex)
            state[0] = 1.0

        # Apply each gate in sequence using full matrix multiplication
        for gate, target_qubits, control_qubits in self.gates:
            full_gate = self._expand_gate(gate, target_qubits, control_qubits)
            state = full_gate @ state

        return state
