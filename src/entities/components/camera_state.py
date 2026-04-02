from dataclasses import dataclass, field
import numpy as np


@dataclass(slots=True, eq=False)
class CameraState:
    is_panning: bool = False
    is_zooming: bool = False
    is_rotating: bool = False

    pan_speed: float = 0.002
    zoom_speed: float = 0.01
    scroll_zoom_speed: float = 0.1
    rotation_speed: float = 0.2

    focal_point: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))

    front: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    view_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    projection_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    view_projection_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
