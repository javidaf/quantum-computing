import os
import sys
from sklearn.datasets import load_iris
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, AerJob
from qiskit.result import Result

np.random.seed(42)


def main():
    """
    iris = load_iris()
    # print(iris.data)
    # print(iris.target)
    print(iris.feature_names)
    x = iris.data
    y = iris.target
    idx = np.where(y < 2)
    x = x[idx, :]
    y = y[idx]
    print(x.shape, y.shape)
    """

    p = 2
    circuit = QuantumCircuit(p, p)

    sample = np.random.uniform(size=p)
    # target = np.random.uniform(size=1)

    for feature_idx in range(p):
        angle = float(2 * np.pi * sample[feature_idx])
        qubit_idx = int(feature_idx)

        circuit.h(qubit_idx)
        circuit.rz(angle, qubit_idx)

    circuit.measure(0, 0)
    # circuit.measure(1, 1)
    print(circuit)

    simulator = AerSimulator()
    transpiled_circuit = transpile(circuit, simulator)
    job: AerJob = simulator.run(transpiled_circuit, shots=1000)
    results: Result = job.result()
    counts = results.get_counts(transpiled_circuit)

    print(circuit.draw())

    predictions = 0
    for key, value in counts.items():
        if key == "01":
            predictions += value
    predictions /= 1000

    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
