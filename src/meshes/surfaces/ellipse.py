import numpy as np
import math


def generate_ellipse(radius_x=5.0, radius_z=3.0, sides=32):
    """
    Generates an ellipse on the XZ plane.

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    data = []

    data.extend([0.0, 0.0, 0.0, 0, 1, 0, 0.5, 0.5])
    for i in range(sides):
        a = i * 2 * math.pi / sides
        x = radius_x * math.cos(a)
        z = radius_z * math.sin(a)
        data.extend([x, 0.0, z, 0, 1, 0, (x/(2*radius_x)) +
                    0.5, (z/(2*radius_z)) + 0.5])

    offset = sides + 1
    data.extend([0.0, 0.0, 0.0, 0, -1, 0, 0.5, 0.5])
    for i in range(sides):
        a = i * 2 * math.pi / sides
        x = radius_x * math.cos(a)
        z = radius_z * math.sin(a)
        data.extend([x, 0.0, z, 0, -1, 0, (x/(2*radius_x)) +
                    0.5, (z/(2*radius_z)) + 0.5])

    indices = []
    for i in range(1, sides + 1):
        nxt = i + 1 if i < sides else 1
        indices.extend([0, i, nxt])
        indices.extend([offset, offset + nxt, offset + i])

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)
