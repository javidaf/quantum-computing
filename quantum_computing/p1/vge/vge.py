import numpy as np
from scipy.optimize import minimize
from quantum_computing.p1.quantum.circuits import QuantumCircuit
from quantum_computing.p1.quantum.gates import RY, RZ, CNOT, PauliX, PauliY, PauliZ
from quantum_computing.p1.quantum.measurements import measure_expectation
from quantum_computing.p1.quantum.states import Qubit
from quantum_computing.p1.quantum.utils import tensor_product


class VQE:
    """
    Variational Quantum Eigensolver for quantum Hamiltonians.
    Finds the ground state energy using a variational approach.
    Supports both one-qubit and two-qubit systems.
    """

    def __init__(self, hamiltonian=None, num_qubits=1, layers=1, pauli_terms=None):
        """
        Initialize the VQE solver.

        Args:
            hamiltonian: The Hamiltonian matrix (optional if pauli_terms is provided)
            num_qubits: Number of qubits in the system
            layers: Number of variational layers
            pauli_terms: List of tuples (coefficient, [paulis]) where paulis is a list of
                         'I', 'X', 'Y', 'Z' for each qubit (optional)
        """
        self.num_qubits = num_qubits
        self.layers = layers
        self.optimal_params = None
        self.optimal_energy = None
        self.energy_history = []

        # Initialize Pauli operators for convenience
        self.I = np.eye(2, dtype=complex)
        self.X = PauliX().matrix
        self.Y = PauliY().matrix
        self.Z = PauliZ().matrix

        # Set up Hamiltonian representation
        self.pauli_terms = pauli_terms
        self.hamiltonian = hamiltonian

        # If we have a matrix Hamiltonian but no Pauli decomposition,
        # we can decompose it (for 1 or 2 qubits)
        if hamiltonian is not None and pauli_terms is None:
            self.pauli_terms = self.decompose_hamiltonian_to_pauli(
                hamiltonian, num_qubits
            )

        # Calculate number of parameters based on circuit structure
        # For each qubit: 2 parameters (RY, RZ) per layer
        if num_qubits == 1:
            self.num_params = 2 * layers
        elif num_qubits == 2:
            # 2 params per qubit per layer + entangling after each layer
            self.num_params = 4 * layers + layers
        else:
            raise ValueError("Only 1 or 2 qubits are currently supported")

    def decompose_hamiltonian_to_pauli(self, hamiltonian, num_qubits):
        """
        Decompose a Hamiltonian matrix into Pauli terms.

        Args:
            hamiltonian: The Hamiltonian matrix to decompose
            num_qubits: Number of qubits

        Returns:
            List of tuples (coefficient, [paulis])
        """
        pauli_terms = []

        if num_qubits == 1:
            # For 1-qubit, decompose as: H = a*I + b*X + c*Y + d*Z
            a = 0.5 * np.trace(hamiltonian)
            b = 0.5 * np.trace(hamiltonian @ self.X)
            c = 0.5 * np.trace(hamiltonian @ self.Y)
            d = 0.5 * np.trace(hamiltonian @ self.Z)

            if abs(a) > 1e-10:
                pauli_terms.append((float(a.real), ["I"]))
            if abs(b) > 1e-10:
                pauli_terms.append((float(b.real), ["X"]))
            if abs(c) > 1e-10:
                pauli_terms.append((float(c.real), ["Y"]))
            if abs(d) > 1e-10:
                pauli_terms.append((float(d.real), ["Z"]))

        elif num_qubits == 2:
            # For 2-qubits, we need to decompose using all 16 combinations of Paulis
            paulis = ["I", "X", "Y", "Z"]
            matrices = [self.I, self.X, self.Y, self.Z]

            for i, p1 in enumerate(paulis):
                for j, p2 in enumerate(paulis):
                    # Create tensor product P1 ⊗ P2
                    P = np.kron(matrices[i], matrices[j])

                    # Calculate coefficient: coeff = Tr(H·P)/4
                    coeff = np.trace(hamiltonian @ P) / 4.0

                    # Add term if coefficient is significant
                    if abs(coeff) > 1e-10:
                        pauli_terms.append((float(coeff.real), [p1, p2]))

        return pauli_terms

    def create_circuit(self, params):
        """
        Create a parameterized quantum circuit with the given parameters.

        Args:
            params: List of circuit parameters

        Returns:
            QuantumCircuit: Prepared circuit with applied gates
        """
        if len(params) != self.num_params:
            raise ValueError(
                f"Expected {self.num_params} parameters but got {len(params)}"
            )

        # Create quantum circuit
        circuit = QuantumCircuit(self.num_qubits)

        # Initialize in |0...0⟩ state
        if self.num_qubits == 1:
            circuit.initialize(Qubit.ket_0())
        else:
            # Initialize to |00⟩ state for two qubits
            initial_state = np.zeros(2**self.num_qubits, dtype=complex)
            initial_state[0] = 1.0
            circuit.initialize(initial_state)

        param_idx = 0

        # Apply variational layers
        for layer in range(self.layers):
            # Rotation gates for each qubit
            for qubit in range(self.num_qubits):
                circuit.add_gate(RY(params[param_idx]), qubit)
                param_idx += 1
                circuit.add_gate(RZ(params[param_idx]), qubit)
                param_idx += 1

            # Add entangling gates for two-qubit system
            if self.num_qubits == 2 and layer < self.layers:
                # CNOT with control on first qubit, target on second qubit
                circuit.add_gate(CNOT(), 1, 0)
                param_idx += 1

        return circuit

    def cost_function(self, params):
        """
        Calculate expectation value of Hamiltonian for given circuit parameters.

        Args:
            params: Circuit parameters

        Returns:
            float: Expectation value <ψ|H|ψ>
        """
        circuit = self.create_circuit(params)
        state = circuit.simulate()

        # Calculate the expectation value
        if self.hamiltonian is not None and self.pauli_terms is None:
            # Use matrix representation
            expectation = measure_expectation(state, self.hamiltonian)
        else:
            # Use Pauli decomposition
            expectation = self.measure_pauli_expectation(state)

        self.energy_history.append(expectation.real)
        return expectation.real

    def measure_pauli_expectation(self, state):
        """
        Calculate expectation value from Pauli terms.

        Args:
            state: Quantum state vector

        Returns:
            float: Expectation value of the Hamiltonian
        """
        expectation = 0.0

        for coeff, paulis in self.pauli_terms:
            if self.num_qubits == 1:
                # For single qubit, measure each Pauli operator directly
                if paulis[0] == "I":
                    # Identity always gives expectation 1
                    expectation += coeff
                elif paulis[0] == "X":
                    expectation += coeff * measure_expectation(state, self.X)
                elif paulis[0] == "Y":
                    expectation += coeff * measure_expectation(state, self.Y)
                elif paulis[0] == "Z":
                    expectation += coeff * measure_expectation(state, self.Z)

            elif self.num_qubits == 2:
                # For two qubits, we need to calculate tensor products
                p1, p2 = paulis

                # Map strings to matrices
                m1 = {"I": self.I, "X": self.X, "Y": self.Y, "Z": self.Z}[p1]
                m2 = {"I": self.I, "X": self.X, "Y": self.Y, "Z": self.Z}[p2]

                # Calculate tensor product and measure expectation
                operator = np.kron(m1, m2)
                expectation += coeff * measure_expectation(state, operator)

        return expectation

    def optimize(self, initial_params=None, max_iter=100, method="BFGS", attempts=3):
        """
        Run the optimization to find the minimum eigenvalue.

        Args:
            initial_params: Initial parameters (random if None)
            max_iter: Maximum number of iterations
            method: Optimization method to use (default: 'BFGS')
            attempts: Number of optimization attempts with different starting points

        Returns:
            tuple: (optimal parameters, minimum eigenvalue)
        """
        best_energy = float("inf")
        best_params = None
        best_history = []

        for _ in range(attempts):
            if initial_params is None:
                start_params = np.random.random(self.num_params) * 2 * np.pi
            else:
                start_params = initial_params.copy()

            self.energy_history = []

            result = minimize(
                self.cost_function,
                start_params,
                method=method,
                options={"maxiter": max_iter},
            )

            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
                best_history = self.energy_history.copy()

        self.optimal_params = best_params
        self.optimal_energy = best_energy
        self.energy_history = best_history

        return self.optimal_params, self.optimal_energy

    def get_optimal_state(self):
        """
        Return the optimal quantum state after optimization.

        Returns:
            array-like: The state vector for the optimal parameters
        """
        if self.optimal_params is None:
            raise ValueError("Optimization has not been performed yet")

        circuit = self.create_circuit(self.optimal_params)
        return circuit.simulate()

    def get_energy_history(self):
        """
        Return the energy history during optimization.

        Returns:
            list: List of energy values during optimization
        """
        return self.energy_history

    def compute_excited_states(self, num_states=1):
        """
        Compute excited states using orthogonality constraints.
        This is a simple implementation that works for small systems.

        Args:
            num_states: Number of excited states to compute (default: 1)

        Returns:
            list: List of (energy, state) pairs for excited states
        """
        if self.num_qubits > 1:
            raise NotImplementedError(
                "Excited states computation only supports one-qubit systems"
            )

        # Compute ground state first
        if self.optimal_params is None:
            self.optimize()

        ground_state = self.get_optimal_state()
        ground_energy = self.optimal_energy

        results = [(ground_energy, ground_state)]

        # For 2x2 matrices, we can just compute the other eigenstate directly
        if self.hamiltonian.shape == (2, 2):
            # The other state must be orthogonal to the ground state
            # For a 2D Hilbert space, there's only one possibility
            other_state = np.array([-ground_state[1], ground_state[0]], dtype=complex)
            other_energy = measure_expectation(other_state, self.hamiltonian).real
            results.append((other_energy, other_state))

        return results
