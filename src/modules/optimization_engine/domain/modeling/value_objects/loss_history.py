from pydantic import BaseModel, Field


class LossHistory(BaseModel):
    """
    Canonical shape for loss history returned by trainers.
    """

    bin_type: str = ""
    bins: list[float] = Field(default_factory=list)
    n_train: list[int] = Field(default_factory=list)
    train_loss: list[float] = Field(default_factory=list)
    val_loss: list[float] = Field(default_factory=list)
    test_loss: list[float] = Field(default_factory=list)

    class Config:
        extra = "forbid"
