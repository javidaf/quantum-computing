import numpy as np
from collections import Counter


class QuantumMeasurement:
    """Class for performing quantum measurements"""

    @staticmethod
    def measure_qubit(state, qubit_index=0, num_qubits=1):
        """
        Measure a specific qubit in a quantum state

        Parameters:
        -----------
        state : array-like
            Quantum state vector
        qubit_index : int
            Index of qubit to measure (0-indexed)
        num_qubits : int
            Total number of qubits in the system

        Returns:
        --------
        int : Measurement result (0 or 1)
        array-like : Post-measurement state
        """
        probabilities = np.abs(state) ** 2
        dim = 2**num_qubits

        # Group probabilities based on the measured qubit being 0 or 1
        prob_0 = 0
        prob_1 = 0

        for i in range(dim):
            # Check if qubit_index is 0 or 1 in this basis state
            if (i >> qubit_index) & 1:
                prob_1 += probabilities[i]
            else:
                prob_0 += probabilities[i]

        # Perform measurement with appropriate probabilities
        result = 0 if np.random.random() < prob_0 else 1

        # Create post-measurement state
        new_state = np.zeros_like(state)
        norm = 0

        for i in range(dim):
            is_one = (i >> qubit_index) & 1
            if (result == 1 and is_one) or (result == 0 and not is_one):
                new_state[i] = state[i]
                norm += probabilities[i]

        # Normalize the post-measurement state
        if norm > 0:
            new_state /= np.sqrt(norm)

        return result, new_state

    @staticmethod
    def measure_all(state, num_samples=1):
        """
        Perform multiple measurements on the full quantum state

        Parameters:
        -----------
        state : array-like
            Quantum state vector
        num_samples : int
            Number of measurements to perform

        Returns:
        --------
        Counter : Dictionary-like object with measurement outcomes and their counts
        """
        probabilities = np.abs(state) ** 2
        num_states = len(probabilities)
        num_qubits = int(np.log2(num_states))

        # Generate samples according to the probability distribution
        outcomes = np.random.choice(num_states, size=num_samples, p=probabilities)

        # Convert to binary representation
        binary_outcomes = [format(outcome, f"0{num_qubits}b") for outcome in outcomes]

        return Counter(binary_outcomes)


def measure_expectation(state, operator):
    """
    Measure the expectation value of an operator for a given state.

    Args:
        state: Quantum state vector
        operator: Operator matrix

    Returns:
        float: Expectation value <ψ|O|ψ>
    """

    expectation = np.vdot(state, np.dot(operator, state))
    return expectation.real
