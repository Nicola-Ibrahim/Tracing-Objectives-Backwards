from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...dtos import NormalizerConfig


class ProcessDatasetCommand(BaseModel):
    """
    Orchestrates: load raw data -> split+normalize -> save processed .pkl

    source_filename:  raw file key/name used by your raw repository
    dest_filename:    output file key/name to save in processed repo (no extension)
    normalizer_config: config passed to NormalizerFactory.create(...)
    test_size:        0..1
    random_state:     reproducible split
    include_original: if True, attach pareto_set/front to the saved bundle
    overwrite:        if False, raise if file exists
    dataset_source:   which arrays to use for training split
                      'historical' -> (historical_objectives, historical_solutions)
                      'pareto'     -> (pareto_front, pareto_set)
    """

    source_filename: str
    dest_filename: str

    normalizer_config: NormalizerConfig = Field(
        default_factory=lambda: NormalizerConfig(
            type="MinMaxScaler", params={"feature_range": (0, 1)}
        )
    )
    test_size: float = 0.2
    random_state: int = 42
    include_original: bool = True
    overwrite: bool = True
