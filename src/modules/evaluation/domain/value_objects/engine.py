from pydantic import BaseModel

from ..enums.engine_capability import EngineCapability
from ..enums.mapping_direction import MappingDirection


class Engine(BaseModel):
    """
    The stochastic inverse engine being evaluated.
    Capability determines which decision space assessment applies.
    """

    type: str
    version: int
    mapping_direction: MappingDirection
    capability: EngineCapability
