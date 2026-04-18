from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class DirectionalLight:
    color: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
    rotation: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([-90.0, 0.0, 0.0], dtype=np.float32))
    strength: np.float32 = np.float32(1.0)

    enabled: bool = True
    casts_shadow: bool = True

    shadow_map_fbo: int = 0
    shadow_map_texture: int = 0
    light_space_matrices: list[npt.NDArray[np.float32]] = field(default_factory=list)
    cascade_distances: list[float] = field(default_factory=list)
