from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from quantum_computing.p2.qml.quantum_model import QuantumClassifier


def plot_data(X, y, title="Data Visualization", xlabel="Feature 1", ylabel="Feature 2"):
    """Plot 2D data points with different colors for each class."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k", s=100)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="Class")
    plt.grid()


def plot_probability_map(
    model: QuantumClassifier,
    X_data: np.ndarray,
    feature_indices: Tuple[int, int] = (0, 1),
    y_data_scatter: Optional[np.ndarray] = None,
    num_points: int = 50,
    padding_factor: float = 0.1,
    ax=None,
    fig=None,
    show_decision_boundary: bool = True,  # New parameter
    decision_boundary_level: float = 0.5,  # New parameter
) -> Tuple[plt.Axes, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Creates and plots a 2D probability map for two selected features of a QuantumClassifier.
    Other features are fixed at their mean value from X_data.

    Args:
        model: The trained QuantumClassifier instance.
        X_data: The dataset (n_samples, n_features) used to infer feature ranges
                and mean values for non-plotted features.
        feature_indices: Tuple of two integers for the features to plot on x and y axes.
                         Defaults to (0, 1).
        y_data_scatter: Optional array of target labels for X_data. If provided,
                        X_data points are scattered on the plot, colored by these labels.
        num_points: Number of points for each dimension of the grid. Defaults to 50.
        padding_factor: How much to extend the feature range beyond min/max of X_data.
                        Defaults to 0.1.
        ax: Matplotlib axis to plot on. If None, a new figure and axis are created.
        fig: Matplotlib figure. Used with ax. If None and ax is None, a new figure is created.
        show_decision_boundary: If True, plots the decision boundary (contour at 0.5 probability).
                                Defaults to True.
        decision_boundary_level: The probability level for the decision boundary.
                                 Defaults to 0.5.


    Returns:
        Tuple containing:
            - ax: The Matplotlib axis with the plot.
            - (X_grid, Y_grid, Z_grid): The grid coordinates and corresponding probabilities.
    """
    if X_data.shape[1] < 2:
        raise ValueError("X_data must have at least 2 features to create a 2D map.")
    if not (
        0 <= feature_indices[0] < X_data.shape[1]
        and 0 <= feature_indices[1] < X_data.shape[1]
    ):
        raise ValueError(
            f"feature_indices {feature_indices} are out of bounds for data with {X_data.shape[1]} features."
        )
    if feature_indices[0] == feature_indices[1]:
        raise ValueError("feature_indices must specify two different features.")

    _n_model_features = model.num_qubits

    feat1_min, feat1_max = (
        X_data[:, feature_indices[0]].min(),
        X_data[:, feature_indices[0]].max(),
    )
    feat2_min, feat2_max = (
        X_data[:, feature_indices[1]].min(),
        X_data[:, feature_indices[1]].max(),
    )

    range1_span = feat1_max - feat1_min
    range2_span = feat2_max - feat2_min
    padding1 = range1_span * padding_factor if range1_span > 1e-9 else padding_factor
    padding2 = range2_span * padding_factor if range2_span > 1e-9 else padding_factor

    x_plot_range = np.linspace(feat1_min - padding1, feat1_max + padding1, num_points)
    y_plot_range = np.linspace(feat2_min - padding2, feat2_max + padding2, num_points)
    X_grid, Y_grid = np.meshgrid(x_plot_range, y_plot_range)

    if X_data.shape[1] != _n_model_features:
        print(
            f"Warning: X_data has {X_data.shape[1]} features, but model expects {_n_model_features}. "
            "Ensure feature_indices correctly map to the model's input features "
            "or that X_data is preprocessed to match model's input dimension."
        )

        if X_data.shape[1] < _n_model_features:
            raise ValueError(
                f"X_data has {X_data.shape[1]} features, which is less than model's num_qubits ({_n_model_features})."
            )

        base_sample_source = X_data[:, :_n_model_features]
        base_sample = np.mean(base_sample_source, axis=0)

    else:
        base_sample = np.mean(X_data, axis=0)

    Z_grid = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            current_sample_for_model = base_sample.copy()
            if (
                feature_indices[0] >= _n_model_features
                or feature_indices[1] >= _n_model_features
            ):
                raise ValueError(
                    f"feature_indices {feature_indices} are out of bounds for model with {_n_model_features} features."
                )

            current_sample_for_model[feature_indices[0]] = X_grid[i, j]
            current_sample_for_model[feature_indices[1]] = Y_grid[i, j]

            Z_grid[i, j] = model.predict_single(current_sample_for_model)

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            ax = fig.add_subplot(111)

    contour = ax.contourf(
        X_grid, Y_grid, Z_grid, levels=np.linspace(0, 1, 11), cmap="RdYlBu", alpha=0.9
    )
    if fig:
        fig.colorbar(contour, ax=ax, label="Prediction Probability (Class 1)")

    if show_decision_boundary:
        ax.contour(
            X_grid,
            Y_grid,
            Z_grid,
            levels=[decision_boundary_level],
            colors="black",
            linewidths=1.5,
            linestyles="--",
        )

    if y_data_scatter is not None:
        ax.scatter(
            X_data[:, feature_indices[0]],
            X_data[:, feature_indices[1]],
            c=y_data_scatter,
            cmap="RdYlBu",
            edgecolors="k",
            marker="o",
            s=70,
            alpha=0.5,
            label="Data Points",
        )
        ax.legend()

    ax.set_xlabel(f"Feature {feature_indices[0]}")
    ax.set_ylabel(f"Feature {feature_indices[1]}")
    ax.set_title(f"Model Probability Map")

    return ax, (X_grid, Y_grid, Z_grid)
