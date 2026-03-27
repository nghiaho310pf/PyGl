from dataclasses import dataclass
from enum import Enum


class DrawMode(Enum):
    Normal = 0
    Wireframe = 1
    DepthOnly = 2


@dataclass
class RenderState:
    target_camera: int | None = None
    draw_mode: DrawMode = DrawMode.Normal
