from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class Transform:
    position: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    rotation: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    scale: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
