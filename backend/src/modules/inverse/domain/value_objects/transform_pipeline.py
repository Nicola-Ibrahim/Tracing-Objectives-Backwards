import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ....modeling.domain.interfaces.base_transform import BaseTransformer


class TransformPipeline(BaseModel):
    """
    Ordered sequence of fitted transforms with domain-aware operations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    transforms: list[tuple[str, BaseTransformer]] = Field(default_factory=list)

    def get_objectives_transforms(self) -> list[BaseTransformer]:
        return [t for label, t in self.transforms if label == "objectives"]

    def get_decisions_transforms(self) -> list[BaseTransformer]:
        return [t for label, t in self.transforms if label == "decisions"]

    def transform_objectives(self, data: np.ndarray) -> np.ndarray:
        """Applies internal transforms to an incoming target objective."""
        result = data.copy()
        for t in self.get_objectives_transforms():
            result = t.transform(result)
        return result

    def transform_decisions(self, data: np.ndarray) -> np.ndarray:
        """Applies internal transforms to an incoming decision."""
        result = data.copy()
        for t in self.get_decisions_transforms():
            result = t.transform(result)
        return result

    def detransform_objectives(self, data: np.ndarray) -> np.ndarray:
        """Applies inverse transforms to an incoming objective."""
        if len(data) == 0:
            return data
        result = data.copy()
        for t in reversed(self.get_objectives_transforms()):
            result = t.inverse_transform(result)
        return result

    def detransform_decisions(self, data: np.ndarray) -> np.ndarray:
        """Applies inverse transforms to an incoming decision."""
        if len(data) == 0:
            return data
        result = data.copy()
        for t in reversed(self.get_decisions_transforms()):
            result = t.inverse_transform(result)
        return result
