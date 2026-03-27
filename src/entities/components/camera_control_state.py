from dataclasses import dataclass

import numpy as np


@dataclass
class CameraControlState:
    is_panning: bool = False
    is_zooming: bool = False
    is_rotating: bool = False

    pan_speed: float = 0.002
    zoom_speed: float = 0.01
    scroll_zoom_speed: float = 0.1
    rotation_speed: float = 0.2

    focal_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    focal_point_distance = 5.0
