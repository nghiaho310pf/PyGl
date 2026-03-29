import numpy as np
import math


def generate_star(outer_radius=5.0, inner_radius=2.0, points=5):
    """
    Generates a star shape on the XZ plane.

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    sides = points * 2
    data = []

    data.extend([0.0, 0.0, 0.0, 0, 1, 0, 0.5, 0.5])
    for i in range(sides):
        r = outer_radius if i % 2 == 0 else inner_radius
        a = math.pi/2 + i * 2 * math.pi / sides
        x = r * math.cos(a)
        z = r * math.sin(a)
        data.extend([x, 0.0, z, 0, 1, 0, (x/(2*outer_radius)) +
                    0.5, (z/(2*outer_radius)) + 0.5])

    offset = sides + 1
    data.extend([0.0, 0.0, 0.0, 0, -1, 0, 0.5, 0.5])
    for i in range(sides):
        r = outer_radius if i % 2 == 0 else inner_radius
        a = math.pi/2 + i * 2 * math.pi / sides
        x = r * math.cos(a)
        z = r * math.sin(a)
        data.extend([x, 0.0, z, 0, -1, 0, (x/(2*outer_radius)) +
                    0.5, (z/(2*outer_radius)) + 0.5])

    indices = []
    for i in range(1, sides + 1):
        nxt = i + 1 if i < sides else 1
        indices.extend([0, nxt, i])
        indices.extend([offset, offset + i, offset + nxt])

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)
