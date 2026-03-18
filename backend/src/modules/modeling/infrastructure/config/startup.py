from .di import ModelingContainer


class ModelingStartUp:
    def __init__(self) -> None:
        self._container: ModelingContainer | None = None

    @property
    def container(self) -> ModelingContainer:
        if self._container is None:
            raise RuntimeError("Modeling container not initialized")
        return self._container

    def initialize(self, wires: list[str]) -> "ModelingStartUp":
        self._container = ModelingContainer()
        self._container.init_resources()
        self._container.wire(
            modules=wires,
        )
        return self

    def shutdown(self) -> None:
        if self._container is not None:
            self._container.shutdown_resources()
            self._container = None
