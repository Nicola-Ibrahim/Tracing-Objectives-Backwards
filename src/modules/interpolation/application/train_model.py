from typing import Any

from sklearn.model_selection import train_test_split

from ...generating.domain.entities.pareto_data import ParetoDataModel
from ...shared.adapters.archivers.npz import ParetoNPzArchiver
from ..adapters.visualization import plot_pareto_visualizations
from ..domain.entities.interpolation_model import InterpolatorModel
from ..domain.interfaces.interpolator import BaseInterpolator
from ..domain.interfaces.logger import BaseLogger
from ..domain.services.metrics import ValidationMetricMethod
from ..domain.services.normalizers import HypercubeNormalizer


def train_model(
    interpolator: BaseInterpolator,
    interpolator_name: str,
    interpolator_params: dict[str, Any],
    logger: BaseLogger,
    validation_metric: ValidationMetricMethod,
):
    # Load saved data
    archiver = ParetoNPzArchiver()
    raw_data: ParetoDataModel = archiver.load(filename="pareto_data.npz")

    # Access individual components
    X_data = raw_data.pareto_set
    Y_data = raw_data.pareto_front

    # Split data into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, Y_data, test_size=0.33, random_state=42
    )

    # Create normalizers - using MinMaxScaler equivalent for both X and Y
    x_normalizer = HypercubeNormalizer(
        feature_range=(-1, 1)
    )  # decision values can be negative or positive

    y_normalizer = HypercubeNormalizer(
        feature_range=(0, 1)
    )  # objective values are always positive

    # Normalize training data
    X_train_norm = x_normalizer.fit_transform(X_train)
    Y_train_norm = y_normalizer.fit_transform(y_train)

    # Transform validation data using same parameters
    X_val_norm = x_normalizer.transform(X_val)
    Y_val_norm = y_normalizer.transform(y_val)

    # Create and fit the interpolator
    interpolator.fit(
        candidate_solutions=X_train_norm,
        objective_front=Y_train_norm,
    )

    # Predict normalized values and inverse-transform
    y_pred_norm = interpolator.generate(X_val_norm)
    y_pred = y_normalizer.inverse_transform(y_pred_norm)

    # Evaluate performance
    mse = validation_metric(y_true=y_val, y_pred=y_pred)

    # Log validation metrics
    logger.log_metrics({"mse": mse})

    # Wrap model in a tracking object
    model = InterpolatorModel(
        name=interpolator_name,
        model=interpolator,
        notes=f"Trained with params: {interpolator_params}",
        model_type=interpolator.__class__.__name__,
    )

    # Save model and metadata to W&B
    logger.log_model(
        name=model.name,
        model=interpolator,
        description=f"{interpolator_name} trained on COCO data",
        model_type=model.model_type,
        parameters=interpolator_params,
        metrics={"mse": mse},
        notes=model.notes,
        collection="interpolator-family",
    )

    # Finish W&B run
    logger.finish()

    # # Find best matching solutions
    # candidate_indices = analyzer.recommend_from_top(user_preference, k=2)
    # print(f"Top candidate solutions at indices: {candidate_indices}")

    # # Calculate interpolation position
    # interpolation_alpha = analyzer.calculate_interpolation_parameter(user_preference)
    # print(f"Interpolation position: α={interpolation_alpha:.2f}")

    # # Generate interpolated solution
    # optimized_solution = interpolator(interpolation_alpha)
    # print(f"Optimized solution: {optimized_solution}")

    # # Verify preference alignment
    # alignment_scores = analyzer.calculate_cosine_similarity(user_preference)
    # print(f"Maximum alignment score: {np.max(alignment_scores):.2f}")

    # # --- Geodesic Interpolation ---
    # geodesic_interp = GeodesicInterpolator(pareto_set)
    # geodesic_solution = geodesic_interp(alpha)
    # print(f"Geodesic Interpolation (α={alpha}):", geodesic_solution)

    # # --- Neural Network Interpolation ---
    # nn_interp = NNInterpolator(input_dim=pareto_set.shape[1])
    # alphas_train = np.linspace(0, 1, len(pareto_set))
    # nn_interp.fit(alphas_train, pareto_set, epochs=5000)
    # nn_solution = nn_interp.predict(alpha)
    # print(f"Neural Net Interpolation (α={alpha}):", nn_solution)
