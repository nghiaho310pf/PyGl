from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class GlobalDrawMode(Enum):
    Normal = 0
    Wireframe = 1
    DepthOnly = 2


@dataclass(slots=True, eq=False)
class BoundingBox:
    entity_id: int
    name: str | None
    classification_name: str
    # screen-space min/max [0.0, 1.0]
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclass(slots=True, eq=False)
class RenderState:
    global_draw_mode: GlobalDrawMode = GlobalDrawMode.Normal

    frame_number: int = 0
    is_capture: bool = False
    show_bounding_boxes: bool = False
    bounding_boxes: list[BoundingBox] = field(default_factory=list)
