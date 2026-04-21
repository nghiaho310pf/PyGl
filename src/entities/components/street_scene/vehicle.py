from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt

from math_utils import vec3


@dataclass(slots=True, eq=False)
class Vehicle:
    speed: float = 0.0
    direction: npt.NDArray[np.float32] = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
