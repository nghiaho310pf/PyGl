import math

import numpy as np


def generate_tetrahedron(size=1.0):
    """
    Generates a flat-shaded tetrahedron.

    Returns:
    vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
    indices: np.array (uint32) -> EBO indices
    """
    s = size / math.sqrt(3)
    v = [
        [s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]
    ]
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    
    data = []
    indices = []
    for i, face in enumerate(faces):
        v0, v1, v2 = np.array(v[face[0]]), np.array(v[face[1]]), np.array(v[face[2]])

        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / np.linalg.norm(normal)

        for vert in [v0, v1, v2]:
            data.extend([*vert, *normal, 0.5, 0.5])

        offset = i * 3
        indices.extend([offset, offset + 1, offset + 2])

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)