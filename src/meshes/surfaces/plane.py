import math

import numpy as np


def generate_plane(size=10.0, uv_scale=1.0):
    """
    Generates a flat plane (floor) on the XZ plane.
    Total vertices: 4
    Total indices: 6 (2 triangles)

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    s = size / 2.0

    data = [
        -s, 0.0, -s, 0, 1, 0, 0, uv_scale,
        -s, 0.0, s, 0, 1, 0, 0, 0,
        s, 0.0, s, 0, 1, 0, uv_scale, 0,
        s, 0.0, -s, 0, 1, 0, uv_scale, uv_scale
    ]

    indices = [
        0, 1, 2,
        0, 2, 3
    ]

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)
