from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class GlobalDrawMode(Enum):
    Normal = 0
    Wireframe = 1
    DepthOnly = 2


@dataclass(slots=True, eq=False)
class GraphicsSettings:
    point_shadow_search_samples: int = 4
    point_shadow_samples: int = 8
    directional_shadow_search_samples: int = 8
    directional_shadow_samples: int = 16

    enable_smaa: bool = True


@dataclass(slots=True, eq=False)
class RenderState:
    global_draw_mode: GlobalDrawMode = GlobalDrawMode.Normal

    shadow_blur_depth_sensitivity: np.float32 = np.float32(5.0)
    shadow_blur_normal_threshold: np.float32 = np.float32(0.7)

    viewport_graphics_settings: GraphicsSettings = field(default_factory=lambda: GraphicsSettings(
        point_shadow_search_samples=4,
        point_shadow_samples=8,
        directional_shadow_search_samples=8,
        directional_shadow_samples=16,

        enable_smaa=True
    ))
    capture_graphics_settings: GraphicsSettings = field(default_factory=lambda: GraphicsSettings(
        point_shadow_search_samples=32,
        point_shadow_samples=64,
        directional_shadow_search_samples=32,
        directional_shadow_samples=64,

        enable_smaa=True
    ))
