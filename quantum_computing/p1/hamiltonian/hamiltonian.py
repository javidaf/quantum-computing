import numpy as np
from quantum_computing.p1.quantum.gates import PauliX, PauliY, PauliZ
from quantum_computing.p1.quantum.utils import (
    tensor_product,
    state_to_density_matrix,
    von_neumann_entropy,
)

from scipy import sparse


class Hamiltonian:
    """
    Class for creating and manipulating Hamiltonians for quantum systems.
    """

    def __init__(self):
        """Initialize the Hamiltonian class"""
        # Pauli matrices for reference
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
        self.X = PauliX().matrix
        self.Y = PauliY().matrix
        self.Z = PauliZ().matrix

    def create_H0(self, E1, E2):
        """
        Create the non-interacting Hamiltonian H0

        Parameters:
        -----------
        E1 : float
            Energy of the first state
        E2 : float
            Energy of the second state

        Returns:
        --------
        numpy.ndarray : 2x2 matrix representing H0
        """
        H0 = np.array([[E1, 0], [0, E2]], dtype=complex)
        return H0

    def create_HI(self, V11, V12, V21, V22):
        """
        Create the interaction Hamiltonian HI

        Parameters:
        -----------
        V11 : float
            Interaction element V11
        V12 : float
            Interaction element V12
        V21 : float
            Interaction element V21
        V22 : float
            Interaction element V22

        Returns:
        --------
        numpy.ndarray : 2x2 matrix representing HI
        """
        HI = np.array([[V11, V12], [V21, V22]], dtype=complex)
        return HI

    def create_total_H(self, H0, HI, lambda_val):
        """
        Create the total Hamiltonian H = H0 + λ*HI

        Parameters:
        -----------
        H0 : numpy.ndarray
            Non-interacting Hamiltonian
        HI : numpy.ndarray
            Interaction Hamiltonian
        lambda_val : float
            Interaction strength parameter (between 0 and 1)

        Returns:
        --------
        numpy.ndarray : 2x2 matrix representing total H
        """
        return H0 + lambda_val * HI

    def express_in_pauli_basis(self, H0, HI):
        """
        Express H0 and HI in terms of Pauli matrices

        Parameters:
        -----------
        H0 : numpy.ndarray
            Non-interacting Hamiltonian
        HI : numpy.ndarray
            Interaction Hamiltonian

        Returns:
        --------
        dict : Coefficients for H0 (E and Omega)
        dict : Coefficients for HI (c, omega_z, omega_x)
        """
        # For H0: H0 = E*I + Omega*sigma_z
        E = (H0[0, 0] + H0[1, 1]) / 2
        Omega = (H0[0, 0] - H0[1, 1]) / 2

        # For HI: HI = c*I + omega_z*sigma_z + omega_x*sigma_x
        c = (HI[0, 0] + HI[1, 1]) / 2
        omega_z = (HI[0, 0] - HI[1, 1]) / 2
        omega_x = HI[0, 1]  # = HI[1, 0] for symmetry

        H0_coeffs = {"E": E, "Omega": Omega}
        HI_coeffs = {"c": c, "omega_z": omega_z, "omega_x": omega_x}

        return H0_coeffs, HI_coeffs

    def reconstruct_from_pauli(self, H0_coeffs, HI_coeffs, lambda_val):
        """
        Reconstruct the Hamiltonian from Pauli coefficients

        Parameters:
        -----------
        H0_coeffs : dict
            Coefficients for H0 (E and Omega)
        HI_coeffs : dict
            Coefficients for HI (c, omega_z, omega_x)
        lambda_val : float
            Interaction strength parameter (between 0 and 1)

        Returns:
        --------
        numpy.ndarray : 2x2 matrix representing total H
        """
        # H0 = E*I + Omega*sigma_z
        H0 = H0_coeffs["E"] * self.I + H0_coeffs["Omega"] * self.Z

        # HI = c*I + omega_z*sigma_z + omega_x*sigma_x
        HI = (
            HI_coeffs["c"] * self.I
            + HI_coeffs["omega_z"] * self.Z
            + HI_coeffs["omega_x"] * self.X
        )

        # H = H0 + lambda*HI
        H = H0 + lambda_val * HI

        return H

    def solve_eigenvalue_problem(self, H):
        """
        Solve the eigenvalue problem for the given Hamiltonian

        Parameters:
        -----------
        H : numpy.ndarray
            Hamiltonian matrix

        Returns:
        --------
        numpy.ndarray : Eigenvalues
        numpy.ndarray : Eigenvectors (columns)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        return eigenvalues, eigenvectors

    def eigenvalues_vs_lambda(self, H0, HI, lambda_values):
        """
        Calculate eigenvalues as a function of lambda

        Parameters:
        -----------
        H0 : numpy.ndarray
            Non-interacting Hamiltonian
        HI : numpy.ndarray
            Interaction Hamiltonian
        lambda_values : array-like
            Array of lambda values between 0 and 1

        Returns:
        --------
        list : List of eigenvalues for each lambda value
        list : List of eigenvectors for each lambda value
        """
        eigenvalues_list = []
        eigenvectors_list = []

        for lambda_val in lambda_values:
            H = self.create_total_H(H0, HI, lambda_val)
            eigenvalues, eigenvectors = self.solve_eigenvalue_problem(H)
            eigenvalues_list.append(eigenvalues)
            eigenvectors_list.append(eigenvectors)

        return eigenvalues_list, eigenvectors_list


class TwoQubitHamiltonian:
    """
    Class for creating and analyzing a two-qubit Hamiltonian system as described in Part D.
    The Hamiltonian has the form:
    H = H_0 + λ*H_I
    where H_0 is the non-interacting part and H_I is the interaction part.
    """

    def __init__(self, energies_noninteracting, H_x, H_z):
        """
        Initialize the two-qubit Hamiltonian.

        Parameters:
        -----------
        energies_noninteracting : list
            List of four energies [ε_00, ε_01, ε_10, ε_11] for the non-interacting Hamiltonian
        H_x : float
            Interaction strength parameter for the σ_x⊗σ_x term
        H_z : float
            Interaction strength parameter for the σ_z⊗σ_z term
        """
        # Store parameters
        self.energies = energies_noninteracting
        self.H_x = H_x
        self.H_z = H_z

        # Basic operators
        self.I = np.eye(2, dtype=complex)
        self.X = PauliX().matrix
        self.Z = PauliZ().matrix

        # Create the Hamiltonian matrices
        self.H0 = self._create_H0()
        self.HI = self._create_HI()

    def _create_H0(self):
        """Create the non-interacting Hamiltonian matrix"""
        # H0 corresponds to the diagonal part with the non-interacting energies
        H0 = np.zeros((4, 4), dtype=complex)
        H0[0, 0] = self.energies[0]  # ε_00
        H0[1, 1] = self.energies[2]  # ε_10
        H0[2, 2] = self.energies[1]  # ε_01
        H0[3, 3] = self.energies[3]  # ε_11
        return H0

    def _create_HI(self):
        """Create the interaction Hamiltonian matrix"""
        # H_I = H_x * (σ_x⊗σ_x) + H_z * (σ_z⊗σ_z)
        sigma_x_tensor = tensor_product(self.X, self.X)
        sigma_z_tensor = tensor_product(self.Z, self.Z)

        HI = self.H_x * sigma_x_tensor + self.H_z * sigma_z_tensor
        return HI

    def create_total_H(self, lambda_val):
        """
        Create the total Hamiltonian H = H0 + λ*HI

        Parameters:
        -----------
        lambda_val : float
            Interaction strength parameter (between 0 and 1)

        Returns:
        --------
        numpy.ndarray : 4x4 matrix representing total H
        """
        return self.H0 + lambda_val * self.HI

    def solve_eigenvalue_problem(self, lambda_val):
        """
        Solve the eigenvalue problem for a given lambda value

        Parameters:
        -----------
        lambda_val : float
            Interaction strength parameter (between 0 and 1)

        Returns:
        --------
        numpy.ndarray : Eigenvalues (sorted)
        numpy.ndarray : Eigenvectors (columns)
        """
        H = self.create_total_H(lambda_val)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # Sort by eigenvalues
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def eigenvalues_vs_lambda(self, lambda_values):
        """
        Calculate eigenvalues as a function of lambda

        Parameters:
        -----------
        lambda_values : array-like
            Array of lambda values between 0 and 1

        Returns:
        --------
        list : List of eigenvalues for each lambda value
        """
        eigenvalues_list = []
        eigenvectors_list = []

        for lambda_val in lambda_values:
            eigenvalues, eigenvectors = self.solve_eigenvalue_problem(lambda_val)
            eigenvalues_list.append(eigenvalues)
            eigenvectors_list.append(eigenvectors)

        return eigenvalues_list, eigenvectors_list

    def compute_reduced_density_matrix(self, state_vector, subsystem="A"):
        """
        Compute the reduced density matrix by tracing out one of the subsystems

        Parameters:
        -----------
        state_vector : numpy.ndarray
            The quantum state vector (4 elements for 2 qubits)
        subsystem : str
            Which subsystem to keep ('A' or 'B')

        Returns:
        --------
        numpy.ndarray : 2x2 reduced density matrix
        """
        # First create the full density matrix
        rho = state_to_density_matrix(state_vector)

        # Perform partial trace to get reduced density matrix
        rho_reduced = np.zeros((2, 2), dtype=complex)

        if subsystem == "A":
            # Trace out subsystem B (second qubit)
            rho_reduced[0, 0] = rho[0, 0] + rho[1, 1]  # <0|ρ|0>_B
            rho_reduced[0, 1] = rho[0, 2] + rho[1, 3]  # <0|ρ|1>_B (off-diagonal)
            rho_reduced[1, 0] = rho[2, 0] + rho[3, 1]  # <1|ρ|0>_B (off-diagonal)
            rho_reduced[1, 1] = rho[2, 2] + rho[3, 3]  # <1|ρ|1>_B
        else:
            # Trace out subsystem A (first qubit)
            rho_reduced[0, 0] = rho[0, 0] + rho[2, 2]  # <0|ρ|0>_A
            rho_reduced[0, 1] = rho[0, 1] + rho[2, 3]  # <0|ρ|1>_A (off-diagonal)
            rho_reduced[1, 0] = rho[1, 0] + rho[3, 2]  # <1|ρ|0>_A (off-diagonal)
            rho_reduced[1, 1] = rho[1, 1] + rho[3, 3]  # <1|ρ|1>_A

        return rho_reduced

    def compute_von_neumann_entropy(self, state_vector, subsystem="A"):
        """
        Compute the von Neumann entropy for a subsystem

        Parameters:
        -----------
        state_vector : numpy.ndarray
            The quantum state vector
        subsystem : str
            Which subsystem to analyze ('A' or 'B')

        Returns:
        --------
        float : The von Neumann entropy
        """
        # Get the reduced density matrix
        rho_reduced = self.compute_reduced_density_matrix(state_vector, subsystem)

        # Calculate von Neumann entropy
        return von_neumann_entropy(rho_reduced)

    def entropy_vs_lambda(self, lambda_values, state_index=0, subsystem="A"):
        """
        Calculate entropy as a function of lambda for a specific eigenstate

        Parameters:
        -----------
        lambda_values : array-like
            Array of lambda values between 0 and 1
        state_index : int
            Index of the eigenstate to analyze (0 for ground state)
        subsystem : str
            Which subsystem to analyze ('A' or 'B')

        Returns:
        --------
        list : List of entropy values for each lambda
        """
        entropy_values = []

        for lambda_val in lambda_values:
            _, eigenvectors = self.solve_eigenvalue_problem(lambda_val)
            state = eigenvectors[:, state_index]
            entropy = self.compute_von_neumann_entropy(state, subsystem)
            entropy_values.append(entropy)

        return entropy_values


class LipkinHamiltonian:
    """
    Class for creating and analyzing the Lipkin Hamiltonian model.
    The Lipkin model is a schematic many-body model with a Hamiltonian:
    H = H_0 + H_1 + H_2
    where H_0 = ε*J_z and H_1 = (V/2)*J_+^2 + (V/2)*J_-^2
    and H_2 = (W/2)*J_+*J_- + (W/2)*J_-*J_+
    """

    def __init__(self):
        """Initialize the LipkinHamiltonian class"""
        # Pauli matrices for reference
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
        self.X = PauliX().matrix
        self.Y = PauliY().matrix
        self.Z = PauliZ().matrix

    def _j_plus_operator(self, j):
        """Create the J_+ operator for a given j value"""
        dim = int(2 * j + 1)
        jplus = np.zeros((dim, dim), dtype=complex)

        for m in range(-j, j):
            # Convert to matrix indices (m+j)
            idx = int(m + j)
            # J_+ |j,m⟩ = √(j(j+1) - m(m+1)) |j,m+1⟩
            coeff = np.sqrt(j * (j + 1) - m * (m + 1))
            jplus[idx + 1, idx] = coeff

        return jplus

    def _j_minus_operator(self, j):
        """Create the J_- operator for a given j value"""
        # J_- is the Hermitian conjugate of J_+
        return self._j_plus_operator(j).T.conj()

    def _j_z_operator(self, j):
        """Create the J_z operator for a given j value"""
        dim = int(2 * j + 1)
        jz = np.zeros((dim, dim), dtype=complex)

        for m in range(-j, j + 1):
            # Convert to matrix indices (m+j)
            idx = int(m + j)
            # J_z |j,m⟩ = m |j,m⟩
            jz[idx, idx] = m

        return jz

    def create_j1_hamiltonian(self, epsilon, V, W):
        """
        Create the Lipkin Hamiltonian for J=1 case

        Parameters:
        -----------
        epsilon : float
            Single-particle energy parameter
        V : float
            Interaction strength parameter
        W : float, optional
            Second interaction parameter, default is 0

        Returns:
        --------
        numpy.ndarray : 3x3 matrix representing the J=1 Hamiltonian
        """
        # For J=1, the basis states are |1,-1⟩, |1,0⟩, |1,1⟩
        j = 1

        # Create the operators
        jz = self._j_z_operator(j)
        jplus = self._j_plus_operator(j)
        jminus = self._j_minus_operator(j)

        # Create the Hamiltonian
        # Use the equation 14 in https://journals-aps-org.ezproxy.uio.no/prc/abstract/10.1103/PhysRevC.106.024319
        H0 = epsilon * jz
        H1 = -(V / 2) * (jplus @ jplus + jminus @ jminus)
        H2 = -(W / 2) * (jplus @ jminus + jminus @ jplus)
        return H0 + H1 + H2

    def create_j2_hamiltonian(self, epsilon, V, W):
        """
        Create the Lipkin Hamiltonian for J=2 case

        Parameters:
        -----------
        epsilon : float
            Single-particle energy parameter
        V : float
            Interaction strength parameter
        W : float, optional
            Second interaction parameter, default is 0

        Returns:
        --------
        numpy.ndarray : 5x5 matrix representing the J=2 Hamiltonian
        """
        # For J=2, the basis states are |2,-2⟩, |2,-1⟩, |2,0⟩, |2,1⟩, |2,2⟩
        j = 2

        # Create the operators
        jz = self._j_z_operator(j)
        jplus = self._j_plus_operator(j)
        jminus = self._j_minus_operator(j)

        # Create the Hamiltonian
        # Use the equation 15 in https://journals-aps-org.ezproxy.uio.no/prc/abstract/10.1103/PhysRevC.106.024319
        H0 = epsilon * jz
        H1 = -(V / 2) * (jplus @ jplus + jminus @ jminus)
        H2 = -(W / 2) * (jminus @ jplus + jplus @ jminus)

        return H0 + H1 + H2

    def decompose_j1_to_pauli(self, epsilon, V, W):
        """
        Decompose the J=1 Hamiltonian to Pauli terms.
        Since J=1 is a 3×3 matrix, we need at least 2 qubits to represent it.
        We'll use a specific encoding of the 3 states in a 4-dimensional space.

        Parameters:
        -----------
        epsilon : float
            Single-particle energy parameter
        V : float
            Interaction strength parameter

        Returns:
        --------
        list : List of tuples (coefficient, [paulis]) representing the decomposition
        """
        # Create the J=1 Hamiltonian
        H_j1 = self.create_j1_hamiltonian(epsilon, V, W)

        # To represent a 3×3 matrix with Pauli operators, we need to embed it in
        # a 4×4 space (2 qubits) and find the corresponding Pauli decomposition

        # First, embed the 3×3 matrix in a 4×4 space by padding with zeros
        H_embedded = np.zeros((4, 4), dtype=complex)
        H_embedded[:3, :3] = H_j1

        # Now decompose the embedded Hamiltonian into Pauli terms
        pauli_terms = []
        paulis = ["I", "X", "Y", "Z"]
        matrices = [self.I, self.X, self.Y, self.Z]

        for i, p1 in enumerate(paulis):
            for j, p2 in enumerate(paulis):
                # Create tensor product P1 ⊗ P2
                P = np.kron(matrices[i], matrices[j])

                # Calculate coefficient: coeff = Tr(H·P)/4
                coeff = np.trace(H_embedded @ P) / 4.0

                # Add term if coefficient is significant
                if abs(coeff) > 1e-10:
                    pauli_terms.append((float(coeff.real), [p1, p2]))

        return pauli_terms

    def decompose_j2_to_pauli(self, epsilon, V, W):
        """
        Decompose the J=2 Hamiltonian to Pauli terms.
        Since J=2 is a 5×5 matrix, we need at least 3 qubits to represent it.
        We'll use a specific encoding of the 5 states in an 8-dimensional space.

        Parameters:
        -----------
        epsilon : float
            Single-particle energy parameter
        V : float
            Interaction strength parameter

        Returns:
        --------
        list : List of tuples (coefficient, [paulis]) representing the decomposition
        """
        # Create the J=2 Hamiltonian
        H_j2 = self.create_j2_hamiltonian(epsilon, V, W)

        # To represent a 5×5 matrix with Pauli operators, we need to embed it in
        # an 8×8 space (3 qubits) and find the corresponding Pauli decomposition

        # First, embed the 5×5 matrix in an 8×8 space by padding with zeros
        H_embedded = np.zeros((8, 8), dtype=complex)
        H_embedded[:5, :5] = H_j2

        # Now decompose the embedded Hamiltonian into Pauli terms
        pauli_terms = []
        paulis = ["I", "X", "Y", "Z"]
        matrices = [self.I, self.X, self.Y, self.Z]

        # For 3 qubits, we need to calculate all 64 combinations of Pauli operators
        for i, p1 in enumerate(paulis):
            for j, p2 in enumerate(paulis):
                for k, p3 in enumerate(paulis):
                    # Create tensor product P1 ⊗ P2 ⊗ P3
                    P = np.kron(np.kron(matrices[i], matrices[j]), matrices[k])

                    # Calculate coefficient: coeff = Tr(H·P)/8
                    coeff = np.trace(H_embedded @ P) / 8.0

                    # Add term if coefficient is significant
                    if abs(coeff) > 1e-10:
                        pauli_terms.append((float(coeff.real), [p1, p2, p3]))

        return pauli_terms

    def solve_eigenvalue_problem(self, hamiltonian):
        """
        Solve the eigenvalue problem for the given Hamiltonian

        Parameters:
        -----------
        hamiltonian : numpy.ndarray
            Hamiltonian matrix

        Returns:
        --------
        numpy.ndarray : Eigenvalues (sorted)
        numpy.ndarray : Eigenvectors (columns, sorted by eigenvalues)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

        # Sort by eigenvalues
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def solve_eigen_vs_V(self, j, epsilon, V_values, W):
        """
        Calculate eigenvalues as a function of interaction strength V

        Parameters:
        -----------
        j : int or float
            Total spin value (1 or 2)
        epsilon : float
            Single-particle energy parameter
        V_values : array-like
            Array of V values

        Returns:
        --------
        list : List of eigenvalues for each V value
        """
        eigenvalues_list = []
        eigenstates_list = []

        for V in V_values:
            if j == 1:
                H = self.create_j1_hamiltonian(epsilon, V, W)
            elif j == 2:
                H = self.create_j2_hamiltonian(epsilon, V, W)
            else:
                raise ValueError("j must be either 1 or 2")

            eigenvalues, eigenstates = self.solve_eigenvalue_problem(H)
            eigenvalues_list.append(eigenvalues)
            eigenstates_list.append(eigenstates)

        return eigenvalues_list, eigenstates_list
