from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(slots=True, eq=False)
class CameraState:
    is_panning: bool = False
    is_zooming: bool = False
    is_rotating: bool = False

    pan_speed: float = 0.002
    zoom_speed: float = 0.01
    scroll_zoom_speed: float = 0.1
    rotation_speed: float = 0.2

    focal_point: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    last_camera_id: int | None = None
    euler_buffer: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    last_synced_rotation: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    front: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    up: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    view_matrix: npt.NDArray[np.float32] = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    projection_matrix: npt.NDArray[np.float32] = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    view_projection_matrix: npt.NDArray[np.float32] = field(default_factory=lambda: np.eye(4, dtype=np.float32))
