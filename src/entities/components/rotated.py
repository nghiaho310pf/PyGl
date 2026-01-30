from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class Rotated:
    delta: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
