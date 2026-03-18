from .di import InverseContainer


class InverseStartUp:
    def __init__(self) -> None:
        self._container: InverseContainer | None = None

    @property
    def container(self) -> InverseContainer:
        if self._container is None:
            raise RuntimeError("Inverse container not initialized")
        return self._container

    def initialize(self, wires: list[str]) -> "InverseStartUp":
        self._container = InverseContainer()
        self._container.init_resources()
        self._container.wire(
            modules=wires,
        )
        return self

    def shutdown(self) -> None:
        if self._container is not None:
            self._container.shutdown_resources()
            self._container = None
