import numpy as np
import math


def generate_pentagon(size=10.0):
    """
    Generates a regular pentagon on the XZ plane.
    Total vertices: 10 (5 top, 5 bottom)
    Total indices: 18 (6 triangles)

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    radius = size / 2.0
    angles = [math.pi/2 + i * 2 * math.pi / 5 for i in range(5)]
    
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
        0, 1, 2,   0, 2, 3,   0, 3, 4, # Top
        5, 7, 6,   5, 8, 7,   5, 9, 8  # Bottom
    ]

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)