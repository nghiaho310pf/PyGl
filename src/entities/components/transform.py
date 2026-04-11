from dataclasses import dataclass, field
from math_utils import vec3, quaternion_identity

import numpy as np
import numpy.typing as npt


@dataclass(slots=True, eq=False)
class Transform:
    position: npt.NDArray[np.float32] = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    rotation: npt.NDArray[np.float32] = field(default_factory=quaternion_identity)
    scale: npt.NDArray[np.float32] = field(default_factory=lambda: vec3(1.0, 1.0, 1.0))
