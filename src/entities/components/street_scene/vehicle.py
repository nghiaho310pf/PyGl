from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt

from math_utils import vec3


@dataclass(slots=True, eq=False)
class Vehicle:
    target_speed: float = 0.0
    current_speed: float = 0.0
    acceleration: float = 0.0

    max_acceleration: float = 2.0
    max_braking: float = 5.0

    direction: npt.NDArray[np.float32] = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    lane_id: int = 0
    target_lane_id: int = 0
    lane_change_progress: float = 0.0
    lane_change_speed: float = 0.3 # 1/seconds for full change

    base_rotation: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
