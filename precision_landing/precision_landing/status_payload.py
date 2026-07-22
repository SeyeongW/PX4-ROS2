"""ROS-independent status helpers for the PX4/MAVROS production mission."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from enum import Enum
from numbers import Integral, Real
from typing import Any


def sanitize_json_value(value: Any) -> Any:
    """Return a strict-JSON-safe copy, mapping non-finite numbers to null."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return sanitize_json_value(value.value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): sanitize_json_value(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    if hasattr(value, 'tolist'):
        return sanitize_json_value(value.tolist())
    if hasattr(value, 'item'):
        return sanitize_json_value(value.item())
    raise TypeError(f'unsupported status value type: {type(value).__name__}')


def dumps_strict_json(payload: Mapping[str, Any]) -> str:
    """Serialize without emitting non-standard NaN/Infinity tokens."""
    return json.dumps(
        sanitize_json_value(payload),
        allow_nan=False,
        separators=(',', ':'),
        sort_keys=True,
    )
