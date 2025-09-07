from ...domain.modeling.interfaces.base_forward_decision_mapper import (
    BaseForwardDecisionMapper,
)
from ...domain.modeling.interfaces.base_loss_calculator import (
    BaseLossCalculator,
    to_numpy,
)


class TandemLossWrapper(BaseLossCalculator):
    def __init__(
        self, base_loss: BaseLossCalculator, forward_model: BaseForwardDecisionMapper
    ):
        self._base_loss = base_loss
        self._forward_model = forward_model

    def calculate(self, predicted_decisions, target_objectives) -> float:
        # Convert input decisions to NumPy
        pred_x = to_numpy(predicted_decisions)
        # Use forward model to map X → Y
        pred_y = self._forward_model.predict(pred_x)
        targ_y = to_numpy(target_objectives)
        return self._base_loss.calculate(pred_y, targ_y)
