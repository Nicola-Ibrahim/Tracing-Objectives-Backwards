from ...infrastructure.loss_calculators.elbo import ELBOLossCalculator
from ...infrastructure.loss_calculators.mse import MSELossCalculator
from ...infrastructure.loss_calculators.nll import (
    NegativeLogLikelihoodLossCalculator,
)


class LossCalculatorFactory:
    def create(self, model, strategy: str):
        if strategy == "mse":
            return MSELossCalculator()
        elif strategy == "nll":
            return NegativeLogLikelihoodLossCalculator(model)
        elif strategy == "elbo":
            return ELBOLossCalculator(model)
        else:
            raise ValueError(f"Unknown loss strategy: {strategy}")
