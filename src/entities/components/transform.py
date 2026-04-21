from dataclasses import dataclass, field
from math_utils import vec3, quaternion_identity

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class TransformData:
    position: npt.NDArray[np.float32]
    rotation: npt.NDArray[np.float32]
    scale: npt.NDArray[np.float32]


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

    def __init__(
        self,
        position: npt.NDArray[np.float32] | None = None,
        rotation: npt.NDArray[np.float32] | None = None,
        scale: npt.NDArray[np.float32] | None = None,
        inherit: bool = True
    ):
        self.local = TransformData(
            position=position if position is not None else vec3(0.0, 0.0, 0.0),
            rotation=rotation if rotation is not None else quaternion_identity(),
            scale=scale if scale is not None else vec3(1.0, 1.0, 1.0)
        )
        self.world = TransformData(
            position=self.local.position.copy(),
            rotation=self.local.rotation.copy(),
            scale=self.local.scale.copy()
        )
        self.inherit = inherit
