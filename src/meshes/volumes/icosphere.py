import math

import numpy as np


def generate_icosphere(radius=1.0, subdivisions=3):
    """
    Generates a sphere by subdividing a icosahedron.
    
    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    phi = (1 + math.sqrt(5)) / 2
    verts = [
        np.array([-1,   phi,    0]), np.array([ 1,   phi,   0]), np.array([-1, -phi, 0]), np.array([ 1, -phi, 0]),
        np.array([ 0,    -1,  phi]), np.array([ 0,     1, phi]), np.array([ 0, -1, -phi]), np.array([ 0,  1, -phi]),
        np.array([ phi,   0,   -1]), np.array([ phi,   0,   1]), np.array([-phi, 0, -1]), np.array([-phi, 0,  1])
    ]
    verts = [v / np.linalg.norm(v) for v in verts]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]

    for _ in range(subdivisions):
        new_faces = []
        for tri in faces:
            v1, v2, v3 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            a = (v1 + v2) / 2; a /= np.linalg.norm(a)
            b = (v2 + v3) / 2; b /= np.linalg.norm(b)
            c = (v3 + v1) / 2; c /= np.linalg.norm(c)
            idx = len(verts)
            verts.extend([a, b, c])
            new_faces.extend([(tri[0], idx, idx+2), (tri[1], idx+1, idx), (tri[2], idx+2, idx+1), (idx, idx+1, idx+2)])
        faces = new_faces

    data = []
    for v in verts:
        u = 0.5 + math.atan2(v[2], v[0]) / (2 * math.pi)
        v_uv = 0.5 - math.asin(v[1]) / math.pi
        data.extend([*(v * radius), *v, u, v_uv])
        
    return np.array(data, dtype=np.float32), np.array(np.array(faces).flatten(), dtype=np.uint32)