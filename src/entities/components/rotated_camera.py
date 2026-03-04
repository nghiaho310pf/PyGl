from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class RotatedCamera:
    center: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    distance: np.float32 = field(default_factory=lambda: np.float32(10))
    rotation_delta: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))