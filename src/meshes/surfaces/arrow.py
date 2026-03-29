import numpy as np
import math


def generate_arrow(size=4.0, tail_length=5.0):
    """
    Generates an arrow pointing in the +Z direction.
    Total vertices: 14
    Total indices: 18 (3 triangles per face)

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    head_w = size
    head_l = size
    tail_w = size / 2.0

    z_tail_end = -(head_l + tail_length) / 2.0
    z_head_base = z_tail_end + tail_length
    z_tip = z_head_base + head_l

    positions = [
        (0.0, z_tip),
        (head_w/2, z_head_base),
        (tail_w/2, z_head_base),
        (tail_w/2, z_tail_end),
        (-tail_w/2, z_tail_end),
        (-tail_w/2, z_head_base),
        (-head_w/2, z_head_base)
    ]

    def get_u(x): return (x / head_w) + 0.5
    def get_v(z): return (z - z_tail_end) / (z_tip - z_tail_end)

    data = []
    for x, z in positions:
        data.extend([x, 0.0, z, 0, 1, 0, get_u(x), get_v(z)])
    for x, z in positions:
        data.extend([x, 0.0, z, 0, -1, 0, get_u(x), get_v(z)])

    indices = [
        0, 1, 6,
        5, 3, 4,
        5, 2, 3,

        7, 13, 8,
        12, 11, 10,
        12, 10, 9
    ]

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)
