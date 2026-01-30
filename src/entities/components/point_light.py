from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class PointLight:
    color: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([300.0, 300.0, 300.0], dtype=np.float32))
