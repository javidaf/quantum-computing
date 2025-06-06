from qiskit import QuantumCircuit, transpile
from qiskit.result import Result, Counts
from qiskit_aer import AerSimulator
import numpy as np
from typing import Optional
from .encoders import BaseEncoder, AngleEncoder
from .ansatzes import BaseAnsatz, SimpleAnsatz
from .initializers import RandomInitializer, BaseInitializer
from .optimizers import Optimizer, AdamOptimizer


class QuantumClassifier:
    """
    Quantum Machine Learning classifier for binary classification.
    Combines data encoding, parameterized ansatz, and measurement.
    """

    def __init__(
        self,
        num_qubits: int,
        encoder: Optional[BaseEncoder] = None,
        ansatz: Optional[BaseAnsatz] = None,
        initializer: Optional[BaseInitializer] = None,
        optimizer: Optional[Optimizer] = None,
        measurement_qubit: int = 0,
        shots: int = 200,
    ):
        """
        Initialize quantum classifier.

        Args:
            num_qubits: Number of qubits in the quantum circuit
            encoder: Data encoding strategy (default: AngleEncoder)
            ansatz: Parameterized quantum circuit (default: SimpleAnsatz)
            measurement_qubit: Which qubit to measure for prediction
        """
        self.num_qubits = num_qubits
        self.encoder = encoder if encoder is not None else AngleEncoder()
        self.ansatz = ansatz if ansatz is not None else SimpleAnsatz()
        self.initializer = (
            initializer if initializer is not None else RandomInitializer()
        )
        self.measurement_qubit = measurement_qubit

        # Initialize parameters
        self.num_params = self.ansatz.num_parameters(num_qubits)
        self.parameters = self.initializer.initialize(self.num_params)

        if optimizer is None:
            self.optimizer = AdamOptimizer()
        elif isinstance(optimizer, type):
            self.optimizer = optimizer()
        else:
            self.optimizer = optimizer

        self.simulator = AerSimulator()
        self.shots = shots

    def _build_circuit(self, features: np.ndarray) -> QuantumCircuit:
        """
        Build the complete quantum circuit for a single sample.

        Args:
            features: Feature vector for one sample

        Returns:
            Complete quantum circuit
        """
        circuit = QuantumCircuit(self.num_qubits, 1)

        # 1. Encode features
        circuit = self.encoder.encode(circuit, features)

        # 2. Apply parameterized ansatz
        circuit = self.ansatz.apply(circuit, self.parameters)
        # 3. Add measurement
        circuit.measure(self.measurement_qubit, 0)

        return circuit

    def cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted probabilities

        Returns:
            Cross-entropy loss
        """
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return float(loss)

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        y_pred_classes = self.predict_classes(X)
        accuracy = np.mean(y_pred_classes == y)
        return accuracy

    def _compute_gradient_parameter_shift(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient using the parameter shift rule for a batch.

        Args:
            X: Feature matrix for a batch
            y: Target labels for a batch

        Returns:
            Gradient array
        """
        gradient = np.zeros(self.num_params)
        original_params = self.get_parameters()

        for i in range(self.num_params):
            # Forward shift: θᵢ + π/2
            shifted_params_plus = original_params.copy()
            shifted_params_plus[i] += np.pi / 2
            self.set_parameters(shifted_params_plus)
            y_pred_plus = self.predict(X)
            loss_plus = self.cross_entropy_loss(y, y_pred_plus)

            # Backward shift: θᵢ - π/2
            shifted_params_minus = original_params.copy()
            shifted_params_minus[i] -= np.pi / 2
            self.set_parameters(shifted_params_minus)
            y_pred_minus = self.predict(X)
            loss_minus = self.cross_entropy_loss(y, y_pred_minus)

            gradient[i] = (loss_plus - loss_minus) / 2.0

            self.set_parameters(original_params)

        return gradient

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: Optional[int] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        tolerance: float = 1e-6,
        verbose: bool = True,
    ) -> dict:
        """
        Train the quantum classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Size of mini-batches (None = full dataset)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            tolerance: Convergence tolerance for early stopping
            verbose: Whether to print training progress

        Returns:
            Training history dictionary
        """
        if self.optimizer is None:
            raise RuntimeError(
                "Optimizer not set. Please provide an optimizer during initialization."
            )

        self.optimizer.initialize(self.get_parameters())

        loss_history = []
        accuracy_history = []
        val_loss_history = []
        val_accuracy_history = []

        n_samples = X_train.shape[0]
        if batch_size is None or batch_size >= n_samples:
            batch_size = n_samples
            use_batching = False
        else:
            batch_size = int(batch_size)
            use_batching = True

        batch_size = int(batch_size)  # type: ignore

        if verbose:
            print("Starting quantum machine learning training...")
            print("=" * 80)
            print("QUANTUM MACHINE LEARNING TRAINING")
            print("=" * 80)
            print("\nMODEL ARCHITECTURE:")
            print(f" • Number of qubits: {self.num_qubits}")
            print(f" • Number of parameters: {self.num_params}")
            print(f" • Encoder type: {type(self.encoder).__name__}")
            print(f" • Ansatz type: {type(self.ansatz).__name__}")
            print(f" • Measurement qubit: {self.measurement_qubit}")
            print("\nTRAINING CONFIGURATION:")
            print(f" • Optimizer: {type(self.optimizer).__name__}")
            if hasattr(self.optimizer, "learning_rate"):
                print(f" • Learning rate: {self.optimizer.learning_rate}")
            print(f" • Epochs: {epochs}")
            if use_batching:
                print(f" • Batch size: {batch_size}")
            else:
                print(" • Batch processing: Full dataset")
            print(f" • Convergence tolerance: {tolerance}")
            print("-" * 80)

        prev_loss = float("inf")

        for epoch in range(epochs):
            if use_batching:
                # Shuffle the data
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]

                # Mini-batch training
                for i in range(0, n_samples, batch_size):
                    X_batch = X_shuffled[i : i + batch_size]
                    y_batch = y_shuffled[i : i + batch_size]

                    # Compute gradient for the batch
                    gradient = self._compute_gradient_parameter_shift(X_batch, y_batch)

                    # Update parameters using the optimizer
                    current_params = self.get_parameters()
                    new_params = self.optimizer.update(current_params, gradient)
                    self.set_parameters(new_params)
            else:
                # Full dataset processing (no batching)
                gradient = self._compute_gradient_parameter_shift(X_train, y_train)
                current_params = self.get_parameters()
                new_params = self.optimizer.update(current_params, gradient)
                self.set_parameters(new_params)

            # Compute metrics for the training dataset
            y_pred_train = self.predict(X_train)
            current_loss = self.cross_entropy_loss(y_train, y_pred_train)
            current_accuracy = self.compute_accuracy(X_train, y_train)

            loss_history.append(current_loss)
            accuracy_history.append(current_accuracy)

            # Validation metrics
            val_loss_epoch = None
            val_accuracy_epoch = None
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_loss_epoch = self.cross_entropy_loss(y_val, y_val_pred)
                val_accuracy_epoch = self.compute_accuracy(X_val, y_val)
                val_loss_history.append(val_loss_epoch)
                val_accuracy_history.append(val_accuracy_epoch)

            # Print progress
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                log_msg = f"Epoch hello {epoch+1}/{epochs}: Loss = {current_loss:.4f}, Acc = {current_accuracy:.4f}"
                if val_loss_epoch is not None:
                    log_msg += f" | Val Loss = {val_loss_epoch:.4f}, Val Acc = {val_accuracy_epoch:.4f}"
                print(log_msg)

            # Check for convergence
            if abs(prev_loss - current_loss) < tolerance or (val_accuracy_epoch > 0.98):  # type: ignore
                if verbose:
                    print(f"\nConverged at epoch {epoch+1}")
                break
            prev_loss = current_loss

        training_history = {
            "loss": loss_history,
            "accuracy": accuracy_history,
            "val_loss": val_loss_history,
            "val_accuracy": val_accuracy_history,
            "final_parameters": self.get_parameters(),
        }

        if verbose:
            print("-" * 80)
            print("Training completed!")
            if loss_history:
                print(f"Final Training Loss: {loss_history[-1]:.4f}")
                print(f"Final Training Accuracy: {accuracy_history[-1]:.4f}")
            if val_loss_history:
                print(f"Final Validation Loss: {val_loss_history[-1]:.4f}")
                print(f"Final Validation Accuracy: {val_accuracy_history[-1]:.4f}")
            print("=" * 80)

        return training_history

    def predict_single(self, features: np.ndarray) -> float:
        """
        Make prediction for a single sample.

        Args:
            features: Feature vector for one sample

        Returns:
            Prediction probability (between 0 and 1)
        """
        circuit = self._build_circuit(features)

        job = self.simulator.run(circuit, shots=self.shots)
        result: Result = job.result()
        counts = result.get_counts(circuit)

        # Calculate probability of measuring |1⟩
        if isinstance(counts, dict):
            prediction = counts.get("1", 0) / self.shots
        else:
            counts_dict = dict(counts)
            prediction = counts_dict.get("1", 0) / self.shots
        return prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for multiple samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Array of predictions of shape (n_samples,)
        """
        predictions = []
        for sample in X:
            pred = self.predict_single(sample)
            predictions.append(pred)
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        predictions = self.predict(X)
        proba = np.column_stack([1 - predictions, predictions])
        return proba

    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary classes.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            threshold: Decision threshold

        Returns:
            Array of predicted classes (0 or 1)
        """
        predictions = self.predict(X)
        return (predictions > threshold).astype(int)

    def set_parameters(self, parameters: np.ndarray):
        """Set the parameters of the quantum circuit."""
        if len(parameters) != self.num_params:
            raise ValueError(
                f"Expected {self.num_params} parameters, got {len(parameters)}"
            )
        self.parameters = parameters.copy()

    def get_parameters(self) -> np.ndarray:
        """Get the current parameters of the quantum circuit."""
        return self.parameters.copy()

    def get_circuit_depth(self, features: np.ndarray) -> int:
        """Get the depth of the quantum circuit for given features."""
        circuit = self._build_circuit(features)
        return circuit.depth()

    def visualize_circuit(self, features: np.ndarray) -> str:
        """
        Get a string representation of the circuit for visualization.

        Args:
            features: Sample features to build circuit with

        Returns:
            String representation of the circuit
        """
        circuit = self._build_circuit(features)
        return str(circuit.draw())
