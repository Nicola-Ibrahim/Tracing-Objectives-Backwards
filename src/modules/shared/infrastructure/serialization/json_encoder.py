import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np


class DiagnosticsJsonEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle non-standard types in diagnostic reports.
    Specifically handles: numpy arrays, numpy scalars, datetimes, and decimals.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.ndarray, list)):
            return [
                self.default(item)
                if not isinstance(item, (int, float, str, bool, type(None)))
                else item
                for item in obj
            ]
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if hasattr(obj, "item") and callable(obj.item):
            # Handle numpy scalars like np.float64
            return obj.item()

        return super().default(obj)


def serialize_diagnostics(data: Any) -> str:
    """
    Utility to serialize diagnostic data to a JSON string.
    """
    return json.dumps(data, cls=DiagnosticsJsonEncoder)
