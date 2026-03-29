import numpy as np
import math


def generate_triangle(size=10.0):
    """
    Generates an equilateral triangle on the XZ plane.
    Total vertices: 6 (3 top, 3 bottom)
    Total indices: 6 (2 triangles)

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    radius = size / 2.0
    angles = [math.pi/2, math.pi/2 + 2*math.pi/3, math.pi/2 + 4*math.pi/3]

    positions = []
    for a in angles:
        x = radius * math.cos(a)
        z = radius * math.sin(a)
        u = (x / (2*radius)) + 0.5
        v = (z / (2*radius)) + 0.5
        positions.append((x, z, u, v))

    data = []
    for p in positions:
        data.extend([p[0], 0.0, p[1], 0, 1, 0, p[2], p[3]])
    for p in positions:
        data.extend([p[0], 0.0, p[1], 0, -1, 0, p[2], p[3]])

    indices = [
        0, 2, 1,
        3, 4, 5
    ]

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)
