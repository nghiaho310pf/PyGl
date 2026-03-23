from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class PointLight:
    color: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
    strength: np.float32 = np.float32(300.0)
    enabled: bool = True
