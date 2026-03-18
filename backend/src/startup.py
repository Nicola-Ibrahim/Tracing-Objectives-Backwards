from .modules.dataset.infrastructure.config.startup import (
    DatasetStartUp,
)
from .modules.evaluation.infrastructure.config.startup import (
    EvaluationStartUp,
)
from .modules.inverse.infrastructure.config.startup import (
    InverseStartUp,
)
from .modules.modeling.infrastructure.config.startup import (
    ModelingStartUp,
)
from .modules.shared.infrastructure.inspection import discover_modules
from .modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


class BackendStartUp:
    def __init__(self):
        self.logger = CMDLogger(name="backend")
        self.dataset_startup = DatasetStartUp()
        self.evaluation_startup = EvaluationStartUp()
        self.inverse_startup = InverseStartUp()
        self.modeling_startup = ModelingStartUp()

    def initialize_modules(self):
        """
        Initializes all modules and wires cross-module dependencies.
        """
        # Discover all API routers to ensure cross-module injection works in every endpoint
        all_routers = discover_modules("src.api.routers.v1")

        # 1. Initialize all sub-containers with the complete set of routers
        self.dataset_startup.initialize(wires=all_routers)
        self.evaluation_startup.initialize(wires=all_routers)
        self.inverse_startup.initialize(wires=all_routers)
        self.modeling_startup.initialize(wires=all_routers)

        # 2. Provide global dependencies
        self.dataset_startup.container.logger.override(self.logger)
        self.evaluation_startup.container.logger.override(self.logger)
        self.inverse_startup.container.logger.override(self.logger)
        self.modeling_startup.container.logger.override(self.logger)

        # 3. Composition (Cross-Wiring)
        # Modeling needs Dataset repository
        self.modeling_startup.container.dataset_repository.override(
            self.dataset_startup.container.repository
        )

        # Inverse needs Dataset repository
        self.inverse_startup.container.dataset_repository.override(
            self.dataset_startup.container.repository
        )

        # Dataset needs Inverse (for engine repository)
        self.dataset_startup.container.engine_repository.override(
            self.inverse_startup.container.repository
        )

        # Evaluation needs both repositories
        self.evaluation_startup.container.engine_repository.override(
            self.inverse_startup.container.repository
        )
        self.evaluation_startup.container.data_repository.override(
            self.dataset_startup.container.repository
        )
        return self

    async def start(self):
        """Async startup logic (if any module-level async resources are added)."""
        pass

    async def stop(self):
        """Gracefully shuts down all module-level resources."""
        self.dataset_startup.shutdown()
        self.evaluation_startup.shutdown()
        self.inverse_startup.shutdown()
        self.modeling_startup.shutdown()


# Exported singleton instance
backend_startup = BackendStartUp()
