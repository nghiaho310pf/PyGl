from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class PointLight:
    color: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
    strength: np.float32 = np.float32(300.0)
    radius: np.float32 = np.float32(0.05)

    enabled: bool = True
    casts_shadow: bool = True

    shadow_map_fbo: int = 0
    shadow_map_texture: int = 0
    light_view_matrices: list[npt.NDArray[np.float32]] = field(default_factory=list)
