import numpy as np
import math


def generate_circle(size=10.0, sides=32):
    """
    Generates a circle on the XZ plane using a center point and a triangle fan.

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    radius = size / 2.0
    data = []

    data.extend([0.0, 0.0, 0.0, 0, 1, 0, 0.5, 0.5])
    for i in range(sides):
        a = i * 2 * math.pi / sides
        x = radius * math.cos(a)
        z = radius * math.sin(a)
        data.extend([x, 0.0, z, 0, 1, 0, (x/size) + 0.5, (z/size) + 0.5])

    offset = sides + 1
    data.extend([0.0, 0.0, 0.0, 0, -1, 0, 0.5, 0.5])
    for i in range(sides):
        a = i * 2 * math.pi / sides
        x = radius * math.cos(a)
        z = radius * math.sin(a)
        data.extend([x, 0.0, z, 0, -1, 0, (x/size) + 0.5, (z/size) + 0.5])

    indices = []
    for i in range(1, sides + 1):
        nxt = i + 1 if i < sides else 1
        indices.extend([0, i, nxt]) # Top
        indices.extend([offset, offset + nxt, offset + i]) # Bottom (Reversed)

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)