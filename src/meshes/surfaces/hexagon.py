import numpy as np
import math

def generate_hexagon(size=10.0):
    """
    Generates a regular hexagon on the XZ plane.
    Total vertices: 12 (6 top, 6 bottom)
    Total indices: 24 (8 triangles total)

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    radius = size / 2.0
    angles = [i * math.pi / 3 for i in range(6)]

    positions = []
    for a in angles:
        x = radius * math.cos(a)
        z = radius * math.sin(a)
        u = (x / size) + 0.5
        v = (z / size) + 0.5
        positions.append((x, z, u, v))

    data = []
    for p in positions:
        data.extend([p[0], 0.0, p[1], 0, 1, 0, p[2], p[3]])
    for p in positions:
        data.extend([p[0], 0.0, p[1], 0, -1, 0, p[2], p[3]])

    indices = [
        0, 2, 1,
        0, 3, 2,
        0, 4, 3,
        0, 5, 4,

        6, 7, 8,
        6, 8, 9,
        6, 9, 10,
        6, 10, 11
    ]

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)
