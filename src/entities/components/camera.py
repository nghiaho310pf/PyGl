from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    fov: np.float32 = np.float32(45.0)
    near: np.float32 = np.float32(0.1)
    far: np.float32 = np.float32(100.0)
