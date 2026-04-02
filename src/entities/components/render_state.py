from dataclasses import dataclass
from enum import Enum


class GlobalDrawMode(Enum):
    Normal = 0
    Wireframe = 1
    DepthOnly = 2


@dataclass(slots=True, eq=False)
class RenderState:
    global_draw_mode: GlobalDrawMode = GlobalDrawMode.Normal
