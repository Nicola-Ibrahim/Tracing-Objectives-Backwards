from .di import DatasetContainer


class DatasetStartUp:
    def __init__(self) -> None:
        self._container: DatasetContainer | None = None

    @property
    def container(self) -> DatasetContainer:
        if self._container is None:
            raise RuntimeError("Dataset container not initialized")
        return self._container

    def initialize(self, wires: list[str]) -> "DatasetStartUp":
        self._container = DatasetContainer()
        self._container.init_resources()
        self._container.wire(
            modules=wires,
        )
        return self

    def shutdown(self) -> None:
        if self._container is not None:
            self._container.shutdown_resources()
            self._container = None
