import math

import numpy as np


def generate_cube(size=1.0):
    """
    Generates a flat shaded cube.
    Total vertices: 24 = 4 per face * 6 faces
    Total indices: 36 = 6 per face * 6 faces

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    s = size / 2.0

    front = [
        -s, s, s, 0, 0, 1, 0, 1,
        -s, -s, s, 0, 0, 1, 0, 0,
        s, -s, s, 0, 0, 1, 1, 0,
        s, s, s, 0, 0, 1, 1, 1
    ]

    back = [
        s, s, -s, 0, 0, -1, 0, 1,
        s, -s, -s, 0, 0, -1, 0, 0,
        -s, -s, -s, 0, 0, -1, 1, 0,
        -s, s, -s, 0, 0, -1, 1, 1
    ]

    top = [
        -s, s, -s, 0, 1, 0, 0, 1,
        -s, s, s, 0, 1, 0, 0, 0,
        s, s, s, 0, 1, 0, 1, 0,
        s, s, -s, 0, 1, 0, 1, 1
    ]

    bottom = [
        -s, -s, s, 0, -1, 0, 0, 1,
        -s, -s, -s, 0, -1, 0, 0, 0,
        s, -s, -s, 0, -1, 0, 1, 0,
        s, -s, s, 0, -1, 0, 1, 1
    ]

    right = [
        s, s, s, 1, 0, 0, 0, 1,
        s, -s, s, 1, 0, 0, 0, 0,
        s, -s, -s, 1, 0, 0, 1, 0,
        s, s, -s, 1, 0, 0, 1, 1
    ]

    left = [
        -s, s, -s, -1, 0, 0, 0, 1,
        -s, -s, -s, -1, 0, 0, 0, 0,
        -s, -s, s, -1, 0, 0, 1, 0,
        -s, s, s, -1, 0, 0, 1, 1
    ]

    vertices = np.array(front + back + top + bottom + right + left, dtype=np.float32)

    indices = []

    for i in range(6):
        offset = i * 4
        indices.extend([
            offset + 0, offset + 1, offset + 2,
            offset + 0, offset + 2, offset + 3
        ])

    return vertices, np.array(indices, dtype=np.uint32)
