import os
import sys
import argparse
from quantum_computing.p2.utils import data_prepperation, visualization
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Quantum Machine Learning Binary Classification"
    )

    # Data preparation arguments
    parser.add_argument(
        "--n-features",
        type=int,
        default=2,
        help="Number of features to use (default: 2)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test set size ratio (default: 0.3)",
    )

    # Model configuration arguments
    parser.add_argument(
        "--num-qubits", type=int, default=2, help="Number of qubits (default: 2)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of ansatz layers (default: 2)"
    )
    parser.add_argument(
        "--measurement-qubit", type=int, default=0, help="Qubit to measure (default: 0)"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=100,
        help="Number of shots for measurement (default: 100)",
    )

    # Training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.7,
        help="Learning rate for optimizer (default: 0.7)",
    )
    parser.add_argument(
        "--epochs", type=int, default=70, help="Number of training epochs (default: 70)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )

    # Visualization arguments
    parser.add_argument(
        "--no-plots", action="store_true", help="Disable plot visualization"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to files instead of showing",
    )

    # Random seed
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set random seed
    np.random.seed(args.seed)

    X, y = data_prepperation.load_iris_binary()
    X_train, X_test, y_train, y_test = data_prepperation.preprocess_data(
        X, y, n_features=args.n_features, test_size=args.test_size
    )
    print("==" * 30)
    print("Data Preparation Complete")
    print("==" * 30)

    import quantum_computing.p2.qml as qml

    optimizer = qml.AdamOptimizer(learning_rate=args.learning_rate)
    initializer = qml.RandomInitializer()
    encoder = qml.HadRotZRotZZEncoder()
    ansatz = qml.AdvancedAnsatz(num_layers=args.num_layers)
    model = qml.QuantumClassifier(
        num_qubits=args.num_qubits,
        encoder=encoder,
        ansatz=ansatz,
        optimizer=optimizer,
        measurement_qubit=args.measurement_qubit,
        shots=args.shots,
        initializer=initializer,
    )

    quantum_circuit = model.visualize_circuit(X_train[0])
    print("==" * 30)
    print("Quantum Circuit Configuration")
    print("==" * 30)
    print("Model Parameters:")
    print(f"Number of qubits: {model.num_qubits}")
    print(f"Number of parameters: {len(model.get_parameters())}")
    print(f"Encoder: {encoder.__class__.__name__}")
    print(f"Ansatz: {ansatz.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Initializer: {initializer.__class__.__name__}")
    print("Quantum Circuit:")
    print("==" * 30)
    print(quantum_circuit)
    print("==" * 30)

    print("Initiating training...")

    if not args.no_plots:
        visualization.plot_data(X_train, y_train, title="Training Data")
        if args.save_plots:
            plt.savefig("training_data.png")
            plt.close()
        else:
            plt.show(block=False)

    history = model.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("Training complete.")
    print("==" * 30)
    print("Training History:")
    print("==" * 30)

    if not args.no_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(history["loss"], label="Loss")
        plt.plot(history["accuracy"], label="Accuracy")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.title("Training History")
        plt.legend()
        if args.save_plots:
            plt.savefig("training_history.png")
            plt.close()
        else:
            plt.show(block=False)

    print("==" * 30)
    print("Final Training Metrics:")
    print(f"Final Loss: {history['loss'][-1]}")
    print(f"Final Accuracy: {history['accuracy'][-1]}")
    print("==" * 30)

    print("Probability Map Visualization")
    if not args.no_plots:
        plot_data = visualization.plot_probability_map(
            model=model, X_data=X_train, y_data_scatter=y_train
        )
        if args.save_plots:
            plt.savefig("probability_map.png")
            plt.close()
        else:
            plt.show()

    if not args.save_plots:
        plt.close("all")
    print("==" * 30)
    print("Modelling complete.")


if __name__ == "__main__":
    main()
