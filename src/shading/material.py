from typing import Optional
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt

from entities.components.textures_state import Texture
from shading.shader import Shader


@dataclass(slots=True, eq=False)
class Material:
    albedo: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    roughness: np.float32 = field(default_factory=lambda: np.float32(0.0))
    metallic: np.float32 = field(default_factory=lambda: np.float32(0.0))
    reflectance: np.float32 = field(default_factory=lambda: np.float32(0.0))
    translucency: np.float32 = field(default_factory=lambda: np.float32(0.0))
    ao: np.float32 = field(default_factory=lambda: np.float32(0.0))

    albedo_map: Texture | None = None
    # normal_map: Texture | None = None
    # specular_map: Texture | None = None
