from dataclasses import dataclass
from enum import Enum

import numpy as np


class GlobalDrawMode(Enum):
    Normal = 0
    Wireframe = 1
    DepthOnly = 2


@dataclass(slots=True, eq=False)
class RenderState:
    global_draw_mode: GlobalDrawMode = GlobalDrawMode.Normal

    shadow_blur_depth_sensitivity: np.float32 = np.float32(5.0)
    shadow_blur_normal_threshold: np.float32 = np.float32(0.7)

    enable_smaa: bool = True
