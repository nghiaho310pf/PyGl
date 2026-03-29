import numpy as np
import math


def generate_trapezoid(top_width=6.0, bottom_width=10.0, height=8.0):
    """
    Generates an isosceles trapezoid on the XZ plane.
    Total vertices: 8
    Total indices: 12

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    hw_top = top_width / 2.0
    hw_bot = bottom_width / 2.0
    hh = height / 2.0

    max_hw = max(hw_top, hw_bot)

    positions = [
        (-hw_bot, -hh, (-hw_bot/(2*max_hw)) + 0.5, 0.0),
        (hw_bot, -hh, (hw_bot/(2*max_hw)) + 0.5, 0.0),
        (hw_top, hh, (hw_top/(2*max_hw)) + 0.5, 1.0),
        (-hw_top, hh, (-hw_top/(2*max_hw)) + 0.5, 1.0)
    ]

    data = []
    for p in positions:
        data.extend([p[0], 0.0, p[1], 0, 1, 0, p[2], p[3]])
    for p in positions:
        data.extend([p[0], 0.0, p[1], 0, -1, 0, p[2], p[3]])

    indices = [
        0, 1, 2,
        0, 2, 3,
        4, 6, 5,
        4, 7, 6
    ]

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)
