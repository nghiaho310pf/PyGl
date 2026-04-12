from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import numpy.typing as npt


class GizmoMode(Enum):
    Translate = auto()
    Rotate = auto()


class GizmoAxis(Enum):
    NoAxis = auto()
    X = auto()
    Y = auto()
    Z = auto()


@dataclass(slots=True, eq=False)
class GizmoState:
    mode: GizmoMode = GizmoMode.Translate
    active_axis: GizmoAxis = GizmoAxis.NoAxis
    is_dragging: bool = False

    initial_mouse_position: npt.NDArray[np.float32] | None = None
    initial_transform_value: npt.NDArray[np.float32] | None = None

    initial_rotation_angle: float = 0.0
    initial_translation_axis_offset: float = 1.0

    gizmo_size: float = 80.0
