from dataclasses import dataclass


@dataclass
class RenderState:
    target_camera: int | None = None
