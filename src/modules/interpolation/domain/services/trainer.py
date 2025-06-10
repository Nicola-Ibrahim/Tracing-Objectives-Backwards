from ..interfaces.interpolator import BaseInterpolator
from ..interfaces.logger import Base
class Trainer:
    def __init__(
        self, interpolator: BaseInterpolator, logger: WandbLogger, config: dict
    ):
        self.interpolator = interpolator
        self.logger = logger
        self.config = config

    def run(self):
        # Load and normalize data
        ...

        # Train interpolator
        self.interpolator.fit(X_train_norm, Y_train_norm)
        predictions = self.interpolator.generate(X_val_norm)
        predictions = y_normalizer.inverse_transform(predictions)

        # Evaluate
        mse = mean_squared_error(y_val, predictions)
        self.logger.log_metrics({"mse": mse})

        # Save model
        self.logger.log_model(...)
        self.logger.finish()
