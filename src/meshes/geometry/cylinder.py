import math
import numpy as np


def generate_cylinder(radius_bottom=1.0, radius_top=1.0, height=2.0, sectors=32):
    """
    Generates a cylinder.

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    data = []
    indices = []

    dr = radius_bottom - radius_top
    dy = height
    length = math.sqrt(dy * dy + dr * dr)
    n_y = dr / length
    n_xz = dy / length

    for i in range(sectors + 1):
        u = i / sectors
        theta = i * 2.0 * math.pi / sectors
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        nx = n_xz * cos_t
        nz = n_xz * sin_t

        data.extend([
            radius_bottom * cos_t, -height / 2.0, radius_bottom * sin_t,
            nx, n_y, nz,
            u, 0.0
        ])
        data.extend([
            radius_top * cos_t, height / 2.0, radius_top * sin_t,
            nx, n_y, nz,
            u, 1.0
        ])

    for i in range(sectors):
        k = i * 2
        indices.extend([k, k + 1, k + 2])
        indices.extend([k + 1, k + 3, k + 2])

    top_offset = len(data) // 8
    data.extend([0.0, height / 2.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
    for i in range(sectors + 1):
        theta = i * 2.0 * math.pi / sectors
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        data.extend([
            radius_top * cos_t, height / 2.0, radius_top * sin_t,
            0.0, 1.0, 0.0,
            cos_t * 0.5 + 0.5, sin_t * 0.5 + 0.5
        ])
    for i in range(sectors):
        indices.extend([top_offset, top_offset + i + 2, top_offset + i + 1])

    bottom_offset = len(data) // 8
    data.extend([0.0, -height / 2.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.5])
    for i in range(sectors + 1):
        theta = i * 2.0 * math.pi / sectors
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        data.extend([
            radius_bottom * cos_t, -height / 2.0, radius_bottom * sin_t,
            0.0, -1.0, 0.0,
            cos_t * 0.5 + 0.5, sin_t * 0.5 + 0.5
        ])
    for i in range(sectors):
        indices.extend([bottom_offset, bottom_offset +
                       i + 1, bottom_offset + i + 2])

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)
