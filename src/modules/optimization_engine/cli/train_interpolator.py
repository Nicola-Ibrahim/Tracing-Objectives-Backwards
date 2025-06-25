from ..application.interpolation.train_model.dtos import (
    GeodesicInterpolatorParams,
    LinearInterpolatorParams,
    NearestNeighborInterpolatorParams,
    NeuralNetworkInterpolatorParams,
)
from ..application.interpolation.train_model.train_interpolator_command import (
    TrainInterpolatorCommand,
)
from ..application.interpolation.train_model.train_interpolator_handler import (
    TrainInterpolatorCommandHandler,
)
from ..domain.interpolation.entities.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)
from ..domain.services.training_service import DecisionMapperTrainingService
from ..infrastructure.inverse_decision_mappers.factory import (
    InverseDecisionMapperFactory,
)
from ..infrastructure.loggers.wandb_logger import WandbLogger
from ..infrastructure.metrics import MeanSquaredErrorMetric
from ..infrastructure.normalizers import MinMaxScalerNormalizer
from ..infrastructure.repositories.pickle_interpolator_repo import (
    PickleInterpolationModelRepository,
)

# --- 1. Instantiate Concrete Dependencies ---

# Data Repository
pareto_data_repository = PickleInterpolationModelRepository()

# Normalizers
x_normalizer = MinMaxScalerNormalizer(feature_range=(0, 1))
y_normalizer = MinMaxScalerNormalizer(feature_range=(0, 1))

# Interpolator Factory
interpolator_factory = InverseDecisionMapperFactory()

# Logger
# Note: For WandbLogger, ensure you are logged in (wandb login) or it might prompt you.
# For testing without actual WandB connection, you might need a MockLogger or disable WandB.
logger = WandbLogger(
    project="Interpolation-models-Demo",
    entity="your_wandb_entity",
    run_name="Demo_Run_1",
)

# Training Service (This is a domain service, so we instantiate the class directly)
training_service = DecisionMapperTrainingService()

# Validation Metric
validation_metric_calculator = MeanSquaredErrorMetric()

# Model Repository (requires ModelFileHandler)
trained_model_repository = PickleInterpolationModelRepository()


# --- 2. Instantiate the TrainInterpolatorCommandHandler ---
command_handler = TrainInterpolatorCommandHandler(
    pareto_data_archiver=pareto_data_repository,
    x_normalizer=x_normalizer,
    y_normalizer=y_normalizer,
    interpolator_factory=interpolator_factory,
    logger=logger,
    interpolator_trainer=training_service,
    validation_metric_calculator=validation_metric_calculator,
    trained_model_repository=trained_model_repository,
)


# --- 3. Create and Execute TrainInterpolatorCommand ---

print("\n--- Executing TrainInterpolatorCommand for Linear Interpolator ---")

# Define specific parameters for a Linear Interpolator
# If your LinearInterpolator doesn't take specific params, you can use InterpolatorParams()
linear_interpolator_params = LinearInterpolatorParams(
    # e.g., `similarity_metric_name="cosine"` if your LinearInterpolator's __init__ takes it
    # For our simple example, it just takes an optional similarity_metric object.
    # We pass an empty dict for now, and the factory will handle defaults/injections.
)


linear_command = TrainInterpolatorCommand(
    data_file_name="pareto_data",
    interpolator_conceptual_name="My1DLinearMapper",
    type=InverseDecisionMapperType.LINEAR,
    params=linear_interpolator_params,  # Pass the specific params DTO
    test_size=0.2,
    random_state=42,
    description="1D Linear Interpolator trained on synthetic data for demonstration.",
    notes="This is a test run to verify the full dependency injection setup.",
    collection="Demo_1D_Mappers",
    training_data_identifier="synthetic_1d_data_v1",
)

try:
    command_handler.handle(linear_command)
    print("\nLinear Interpolator training command executed successfully.")

    # Demonstrate loading the latest version of this model
    latest_linear_model = trained_model_repository.get_latest_version(
        "My1DLinearMapper"
    )
    print(
        f"\nLoaded latest 'My1DLinearMapper' (ID: {latest_linear_model.id}) trained at {latest_linear_model.trained_at}"
    )

except Exception as e:
    print(f"\nAn error occurred during command execution: {e}")
    import traceback

    traceback.print_exc()

# --- Example of another interpolator type (if you implement it) ---
print(
    "\n--- Executing TrainInterpolatorCommand for Pchip Interpolator (Conceptual) ---"
)

pchip_params = PchipInterpolatorParams(degree=3)  # Example Pchip param

pchip_command = TrainInterpolatorCommand(
    data_source_path=str(dummy_data_path),
    interpolator_conceptual_name="My1DPchipMapper",
    type=InterpolatorType.PCHIP,  # Choose the type
    params=pchip_params,
    test_size=0.2,
    random_state=123,
    description="1D Pchip Interpolator demo.",
    notes="Test for Pchip type.",
    collection="Demo_1D_Mappers",
    training_data_identifier="synthetic_1d_data_v1",
)

try:
    command_handler.handle(pchip_command)
    print("\nPchip Interpolator training command executed successfully.")
except Exception as e:
    print(f"\nAn error occurred during Pchip command execution: {e}")
    import traceback

    traceback.print_exc()
