from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class PointLight:
    color: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([300.0, 300.0, 300.0], dtype=np.float32))
    shadow_map_fbo: int = 0
    shadow_map_texture: int = 0
    light_projection_matrix: npt.NDArray[np.float32] = field(default_factory=lambda: np.identity(4, dtype=np.float32))
    light_view_matrix: npt.NDArray[np.float32] = field(default_factory=lambda: np.identity(4, dtype=np.float32))
