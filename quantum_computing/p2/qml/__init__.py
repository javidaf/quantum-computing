"""
Quantum Machine Learning (QML) Package

This package provides tools for quantum machine learning including:
- Data encoders for mapping classical data to quantum states
- Parameterized quantum ansatzes
- A Quantum classifier
- Optimizers implementing the parameter shift rule

Example usage:
    from qml import QuantumClassifier, AngleEncoder, SimpleAnsatz

    # Create quantum model
    model = QuantumClassifier(
        num_qubits=2,
        encoder=AngleEncoder(),
        ansatz=SimpleAnsatz(),
        optimizer=AdamOptimizer(),
        measurement_qubit=1,
        shots=100
    )

    # Train model
    history = model.train(X_train, y_train)
"""

from .quantum_model import QuantumClassifier
from .encoders import BaseEncoder, AngleEncoder
from .ansatzes import (
    BaseAnsatz,
    SimpleAnsatz,
)
from .optimizers import AdamOptimizer
from .initializers import RandomInitializer

__all__ = [
    # Models
    "QuantumClassifier",
    # Encoders
    "BaseEncoder",
    "AngleEncoder",
    # Ansatzes
    "BaseAnsatz",
    "SimpleAnsatz",
    # Optimizers
    "AdamOptimizer",
    # Initializers
    "RandomInitializer",
]

__version__ = "1.0.0"
__author__ = "Javid Rezai"
