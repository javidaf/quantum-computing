import numpy as np
from scipy.optimize import minimize
from quantum_computing.p1.quantum.circuits import QuantumCircuit
from quantum_computing.p1.quantum.gates import RY, RZ, CNOT, PauliX, PauliY, PauliZ
from quantum_computing.p1.quantum.measurements import measure_expectation
from quantum_computing.p1.quantum.utils import tensor_product
from quantum_computing.p1.hamiltonian.hamiltonian import LipkinHamiltonian


class LipkinVQE:
    """
    Specialized VQE implementation for the Lipkin Hamiltonian.
    Handles both J=1 (3x3 matrix) and J=2 (5x5 matrix) cases.
    """

    def __init__(self, j, epsilon, V=1.0, W=0.0, layers=2):
        """
        Initialize the Lipkin VQE solver.

        Args:
            j: Total spin value (1 or 2)
            epsilon: Single-particle energy parameter
            V: Interaction strength parameter
            W: Second interaction parameter (default=0)
            layers: Number of variational layers
        """
        self.j = j
        self.epsilon = epsilon
        self.V = V
        self.W = W
        self.layers = layers
        self.optimal_params = None
        self.optimal_energy = None
        self.energy_history = []

        self.lipkin = LipkinHamiltonian()

        if j == 1:
            # J=1 case requires 2 qubits (dim=4) to represent 3 states
            self.num_qubits = 2
            self.hamiltonian = self.lipkin.create_j1_hamiltonian(epsilon, V, W)
            self.pauli_terms = self.lipkin.decompose_j1_to_pauli(epsilon, V, W)
        elif j == 2:
            # J=2 case requires 3 qubits (dim=8) to represent 5 states
            self.num_qubits = 3
            self.hamiltonian = self.lipkin.create_j2_hamiltonian(epsilon, V, W)
            self.pauli_terms = self.lipkin.decompose_j2_to_pauli(epsilon, V, W)
        else:
            raise ValueError("j must be either 1 or 2")

        self.I = np.eye(2, dtype=complex)
        self.X = PauliX().matrix
        self.Y = PauliY().matrix
        self.Z = PauliZ().matrix

        # Calculate number of parameters based on circuit structure
        if self.num_qubits == 2:
            # For J=1: 2 params per qubit per layer + entangling gates
            self.num_params = 4 * layers + (layers - 1)
        else:  # num_qubits == 3
            # For J=2: 2 params per qubit per layer + more entangling gates
            self.num_params = 6 * layers + 3 * (layers - 1)

    def create_circuit(self, params):
        """
        Create a parameterized quantum circuit for the Lipkin model.

        Args:
            params: List of circuit parameters

        Returns:
            QuantumCircuit: Prepared circuit with applied gates
        """
        if len(params) != self.num_params:
            raise ValueError(
                f"Expected {self.num_params} parameters but got {len(params)}"
            )
        circuit = QuantumCircuit(self.num_qubits)

        # Initialize to |0...0⟩ state
        initial_state = np.zeros(2**self.num_qubits, dtype=complex)
        initial_state[0] = 1.0
        circuit.initialize(initial_state)

        param_idx = 0

        for layer in range(self.layers):
            # Rotation gates for each qubit
            for qubit in range(self.num_qubits):
                circuit.add_gate(RY(params[param_idx]), qubit)
                param_idx += 1
                circuit.add_gate(RZ(params[param_idx]), qubit)
                param_idx += 1

            # Add entangling gates (except after the last layer)
            if layer < self.layers - 1:
                if self.num_qubits == 2:
                    # For J=1 (2 qubits): single CNOT between qubits
                    circuit.add_gate(CNOT(), 1, 0)  # Target qubit 1, control qubit 0
                    param_idx += 1
                else:  # self.num_qubits == 3
                    # For J=2 (3 qubits): more entangling gates
                    # CNOT between qubit 0 (control) and qubit 1 (target)
                    circuit.add_gate(CNOT(), 1, 0)
                    param_idx += 1

                    # CNOT between qubit 1 (control) and qubit 2 (target)
                    circuit.add_gate(CNOT(), 2, 1)
                    param_idx += 1

                    # CNOT between qubit 2 (control) and qubit 0 (target)
                    # This creates a cycle of entanglement
                    circuit.add_gate(CNOT(), 0, 2)
                    param_idx += 1

        return circuit

    def cost_function(self, params):
        """
        Calculate expectation value of the Lipkin Hamiltonian.

        Args:
            params: Circuit parameters

        Returns:
            float: Expectation value <ψ|H|ψ>
        """
        circuit = self.create_circuit(params)
        state = circuit.simulate()

        # For J=1 and J=2, we need to map the quantum state back to the Lipkin model space
        # We will project the quantum state onto the relevant subspace and renormalize

        if self.j == 1:
            # For J=1, we use only the first 3 basis states of the 2-qubit system
            state_projected = state[:3]
            norm = np.sqrt(np.sum(np.abs(state_projected) ** 2))
            if norm > 1e-10:  # Ensure numerical stability
                state_projected = state_projected / norm
                expectation = measure_expectation(state_projected, self.hamiltonian)
            else:
                # If the state has negligible projection onto the subspace, penalize
                expectation = 1000.0  # Large penalty

        elif self.j == 2:
            # For J=2, we use only the first 5 basis states of the 3-qubit system
            state_projected = state[:5]
            norm = np.sqrt(np.sum(np.abs(state_projected) ** 2))
            if norm > 1e-10:  # Ensure numerical stability
                state_projected = state_projected / norm
                expectation = measure_expectation(state_projected, self.hamiltonian)
            else:
                # If the state has negligible projection onto the subspace, penalize
                expectation = 1000.0  # Large penalty

        self.energy_history.append(expectation.real)
        return expectation.real

    def optimize(self, initial_params=None, max_iter=100, method="BFGS", attempts=5):
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
            array-like: The projected state vector for the optimal parameters
        """
        if self.optimal_params is None:
            raise ValueError("Optimization has not been performed yet")

        circuit = self.create_circuit(self.optimal_params)
        state = circuit.simulate()

        # Project and normalize
        if self.j == 1:
            state_projected = state[:3]
        else:  # j == 2
            state_projected = state[:5]

        norm = np.sqrt(np.sum(np.abs(state_projected) ** 2))
        if norm > 1e-10:
            state_projected = state_projected / norm

        return state_projected

    def get_energy_history(self):
        """
        Return the energy history during optimization.

        Returns:
            list: List of energy values during optimization
        """
        return self.energy_history

    def solve_eigen_vs_V(self, V_values, max_iter=100, method="BFGS", attempts=3):
        """
        Calculate eigenvalues using VQE as a function of interaction strength V.

        Args:
            V_values: Array of V values
            max_iter: Maximum iterations per optimization
            method: Optimization method
            attempts: Optimization attempts per V value

        Returns:
            list: List of lowest eigenvalues and corresponding eigenstates for each V value
        """
        eigenvalues = []
        eigenstates = []

        # Save original V value
        original_V = self.V

        for V in V_values:
            # Update the Hamiltonian for this V value
            self.V = V
            if self.j == 1:
                self.hamiltonian = self.lipkin.create_j1_hamiltonian(
                    self.epsilon, V, self.W
                )
            else:  # j == 2
                self.hamiltonian = self.lipkin.create_j2_hamiltonian(
                    self.epsilon, V, self.W
                )

            # Run VQE optimization
            _, energy = self.optimize(
                max_iter=max_iter, method=method, attempts=attempts
            )

            # Store the eigenvalue and state
            eigenvalues.append(energy)
            eigenstate = self.get_optimal_state()
            eigenstates.append(eigenstate)

            print(f"V = {V:.2f}, Energy = {energy:.6f}")

        # Restore original V value
        self.V = original_V
        if self.j == 1:
            self.hamiltonian = self.lipkin.create_j1_hamiltonian(
                self.epsilon, original_V, self.W
            )
        else:  # j == 2
            self.hamiltonian = self.lipkin.create_j2_hamiltonian(
                self.epsilon, original_V, self.W
            )

        return eigenvalues, eigenstates
