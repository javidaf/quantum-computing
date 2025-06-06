from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import numpy as np


def load_iris_binary():
    """Load iris dataset and prepare for binary classification."""
    print("Loading Iris dataset...")
    iris = load_iris()

    print("Feature names:", iris.feature_names)
    print("Target names:", iris.target_names)

    # Use only first two classes for binary classification
    x = iris.data
    y = iris.target
    idx = np.where(y < 2)  # Only setosa (0) and versicolor (1)

    x = x[idx]
    y = y[idx]

    print(f"Dataset shape: {x.shape}")
    print(f"Target distribution: {np.bincount(y)}")

    return x, y


def preprocess_data(X, y, n_features=2, test_size=0.3, random_state=42):
    """Preprocess data for quantum machine learning."""
    print(f"\nPreprocessing data...")

    # Select first n_features
    X_selected = X[:, :n_features]
    print(f"Using {n_features} features: {X_selected.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalize features to [0, 1] for quantum encoding
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Map to [0, 1] range for quantum gates
    X_train_normalized = (X_train_scaled - X_train_scaled.min()) / (
        X_train_scaled.max() - X_train_scaled.min()
    )
    X_test_normalized = (X_test_scaled - X_train_scaled.min()) / (
        X_train_scaled.max() - X_train_scaled.min()
    )

    print(f"Training set: {X_train_normalized.shape[0]} samples")
    print(f"Test set: {X_test_normalized.shape[0]} samples")

    return X_train_normalized, X_test_normalized, y_train, y_test
