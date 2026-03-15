from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ..enums.transform_type import TransformTypeEnum
from ..interfaces.base_transform import BaseTransformer


class ITransformerFactory(ABC):
    """
    Domain interface for a transformer factory.
    """

    @abstractmethod
    def create(self, config: dict[str, Any]) -> BaseTransformer:
        """Creates a transformer from a configuration dictionary."""
        pass

    @abstractmethod
    def get_transformer_schemas(self) -> list[dict[str, Any]]:
        """Returns schemas for available transformers."""
        pass


class TransformationConfig(BaseModel):
    """
    Domain-level configuration for a single transformation step.
    """

    model_config = ConfigDict(use_enum_values=True, frozen=True)

    type: TransformTypeEnum = Field(
        ..., description="The type of the transform to use."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters specific to the transform type.",
    )
    columns: Optional[list[int]] = Field(
        None,
        description="Indices of columns to transform. If None, all columns are transformed.",
    )


class TransformedData(BaseModel):
    """
    Value object representing the data after transformation chains have been applied.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    X: np.ndarray = Field(..., description="Transformed feature matrix")
    y: np.ndarray = Field(..., description="Transformed target vector/matrix")

    @property
    def X_shape(self) -> tuple:
        return self.X.shape

    @property
    def y_shape(self) -> tuple:
        return self.y.shape


class TransformationDomainService:
    """
    Domain service to coordinate the application of transformation chains to datasets.
    """

    def __init__(self, transformer_factory: ITransformerFactory):
        self._transformer_factory = transformer_factory

    def apply_chains(
        self,
        X: np.ndarray,
        y: np.ndarray,
        x_chain: list[TransformationConfig],
        y_chain: list[TransformationConfig],
    ) -> TransformedData:
        """
        Applies independent transformation chains to X and y.
        Returns a TransformedData value object.
        """
        X_curr = X.copy()
        y_curr = y.copy()

        # Apply X chain
        for i, config in enumerate(x_chain):
            transformer = self._transformer_factory.create(config.model_dump())

            if config.columns:
                if any(c >= X_curr.shape[1] for c in config.columns):
                    raise ValueError(
                        f"Column index out of bounds for X space: {config.columns}"
                    )
                # Hybrid approach: transform specific columns
                X_subset = X_curr[:, config.columns]
                transformer.fit(X_subset)
                X_transformed = transformer.transform(X_subset)
                X_curr[:, config.columns] = X_transformed
            else:
                # Default: transform whole space
                transformer.fit(X_curr)
                X_curr = transformer.transform(X_curr)

        # Apply y chain
        for i, config in enumerate(y_chain):
            transformer = self._transformer_factory.create(config.model_dump())
            # y is typically 1D or has fewer columns
            transformer.fit(y_curr)
            y_curr = transformer.transform(y_curr)

        return TransformedData(X=X_curr, y=y_curr)
