from dataclasses import dataclass, field
from math_utils import vec3, quaternion_identity

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class TransformData:
    position: npt.NDArray[np.float32] = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    rotation: npt.NDArray[np.float32] = field(default_factory=lambda: quaternion_identity())
    scale: npt.NDArray[np.float32] = field(default_factory=lambda: vec3(1.0, 1.0, 1.0))


@dataclass(slots=True, eq=False)
class Transform:
    local: TransformData = field(default_factory=lambda: TransformData(
        position=vec3(0.0, 0.0, 0.0),
        rotation=quaternion_identity(),
        scale=vec3(1.0, 1.0, 1.0)
    ))

    world: TransformData = field(default_factory=lambda: TransformData(
        position=vec3(0.0, 0.0, 0.0),
        rotation=quaternion_identity(),
        scale=vec3(1.0, 1.0, 1.0)
    ))

    inherit: bool = True

    matrix_cache: npt.NDArray[np.float32] = field(default_factory=lambda: np.eye(4, dtype=np.float32))
