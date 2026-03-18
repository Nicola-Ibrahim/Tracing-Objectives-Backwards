from .di import EvaluationContainer


class EvaluationStartUp:
    def __init__(self) -> None:
        self._container: EvaluationContainer | None = None

    @property
    def container(self) -> EvaluationContainer:
        if self._container is None:
            raise RuntimeError("Evaluation container not initialized")
        return self._container

    def initialize(self, wires: list[str]) -> "EvaluationStartUp":
        self._container = EvaluationContainer()
        self._container.init_resources()
        self._container.wire(
            modules=wires,
        )
        return self

    def shutdown(self) -> None:
        if self._container is not None:
            self._container.shutdown_resources()
            self._container = None
