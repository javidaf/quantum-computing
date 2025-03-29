import numpy as np
import matplotlib.pyplot as plt


def tensor_product(*matrices):
    """
    Compute the tensor product of multiple matrices

    Parameters:
    -----------
    *matrices : array-like
        The matrices to compute the tensor product of

    Returns:
    --------
    array-like : The tensor product
    """
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def state_to_density_matrix(state):
    """
    Convert a pure state to its density matrix representation

    Parameters:
    -----------
    state : array-like
        Pure state vector

    Returns:
    --------
    array-like : Density matrix
    """
    state = np.array(state, dtype=complex).reshape(-1, 1)
    return np.dot(state, state.conj().T)


def partial_trace(rho, keep, dims):
    """
    Compute the partial trace of a density matrix

    Parameters:
    -----------
    rho : array-like
        The density matrix
    keep : list
        Indices of subsystems to keep
    dims : list
        Dimensions of each subsystem

    Returns:
    --------
    array-like : The reduced density matrix
    """
    if len(dims) == 2 and dims[0] == 2 and dims[1] == 2:
        # Special case for 2-qubit system
        if keep == [0]:  # Keep first qubit (A), trace out second qubit (B)
            reduced_rho = np.zeros((2, 2), dtype=complex)
            reduced_rho[0, 0] = rho[0, 0] + rho[1, 1]
            reduced_rho[0, 1] = rho[0, 2] + rho[1, 3]
            reduced_rho[1, 0] = rho[2, 0] + rho[3, 1]
            reduced_rho[1, 1] = rho[2, 2] + rho[3, 3]
            return reduced_rho
        elif keep == [1]:  # Keep second qubit (B), trace out first qubit (A)
            reduced_rho = np.zeros((2, 2), dtype=complex)
            reduced_rho[0, 0] = rho[0, 0] + rho[2, 2]
            reduced_rho[0, 1] = rho[0, 1] + rho[2, 3]
            reduced_rho[1, 0] = rho[1, 0] + rho[3, 2]
            reduced_rho[1, 1] = rho[1, 1] + rho[3, 3]
            return reduced_rho

    raise NotImplementedError("General partial trace not implemented yet")


def von_neumann_entropy(rho):
    """
    Calculate the von Neumann entropy of a density matrix

    Parameters:
    -----------
    rho : array-like
        Density matrix

    Returns:
    --------
    float : The von Neumann entropy
    """
    # Calculate eigenvalues of density matrix
    eigenvalues = np.linalg.eigvalsh(rho)

    # Remove very small eigenvalues (numerical errors)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Calculate entropy: -Tr(ρ log₂ ρ) = -∑ᵢ λᵢ log₂ λᵢ
    return -np.sum(eigenvalues * np.log2(eigenvalues))


def print_state(state, label=None):
    """
    Print the quantum state in a readable format

    Parameters:
    -----------
    state : array-like
        The quantum state vector
    label : str, optional
        A label for the state (default is None)
    """
    # Convert to numpy array if not already
    state = np.array(state, dtype=complex)

    # Ensure state is a column vector for consistent display
    state = state.reshape(-1)

    # Format the complex values
    formatted = "["
    for i, val in enumerate(state):
        # Check if imaginary part is effectively zero
        if abs(val.imag) < 1e-10:
            formatted += (
                f"{val.real:.0f}"
                if abs(val.real - round(val.real)) < 1e-10
                else f"{val.real:.4f}"
            )
        else:
            # Format complex numbers nicely
            imag_str = f"{abs(val.imag):.4f}i"
            if val.real == 0:
                formatted += f"{'-' if val.imag < 0 else ''}{imag_str}"
            else:
                formatted += f"{val.real:.4f}{'-' if val.imag < 0 else '+'}{imag_str}"
        if i < len(state) - 1:
            formatted += " "
    formatted += "]ᵀ"  # Added transpose symbol

    # Print with label if provided
    if label:
        print(f"{label} = {state} --> {formatted}")
    else:
        print(f"State = {state} --> {formatted}ᵀ")


def plot_eigenvalues(
    v_values,
    eigenvalues,
    j,
    title=None,
    show_gap=False,
):
    """
    Plot the eigenvalues of the Lipkin Hamiltonian as a function of interaction strength V.

    Parameters:
    -----------
    v_values : array-like
        Array of V values (interaction strength)
    eigenvalues : list of arrays
        List containing eigenvalues or probabilites for each V value
    j : int or float
        Total spin value (1 or 2)
    title : str, optional
        Custom title for the plot
    figsize : tuple, optional
        Size of the figure (width, height)
    show_gap : bool, optional
        If True, also plots the energy gap between ground and first excited state
    save_fig : bool, optional
        If True, saves the figure to a file
    filename : str, optional
        Filename for saving the figure (if save_fig is True)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object for further customization if needed
    """
    eigenvalues_array = np.array(eigenvalues)

    fig, ax = plt.subplots()

    num_states = eigenvalues_array[0].size

    Jz_values = np.arange(-j, j + 1, 1)

    for i in range(num_states):

        state_energies = [evals[i] for evals in eigenvalues]

        if j == 1:
            label = f"State {i} (Jz={Jz_values[i]})"
        elif j == 2:
            label = f"State {i} (Jz={Jz_values[i]})"
        else:
            label = f"State {i}"

        ax.plot(v_values, state_energies, label=label)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Eigenvalues of the Lipkin Hamiltonian (J={j})")

    ax.set_xlabel("Interaction Strength V")
    ax.set_ylabel("Energy")

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if show_gap:

        fig_gap, ax_gap = plt.subplots()

        gap = [evals[1] - evals[0] for evals in eigenvalues]

        ax_gap.plot(v_values, gap, "r-")
        ax_gap.set_title(f"Energy Gap between Ground and First Excited States (J={j})")
        ax_gap.set_xlabel("Interaction Strength V")
        ax_gap.set_ylabel("Energy Gap")
        ax_gap.grid(True, alpha=0.3)

    return fig
