import math
import numpy as np

def generate_torus(radius_main=1.0, radius_tube=0.3, main_segments=32, tube_segments=16):
    """
    Generates a torus.

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    data = []
    indices = []

    for i in range(main_segments + 1):
        u = i / main_segments
        theta = i * 2.0 * math.pi / main_segments
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)

        for j in range(tube_segments + 1):
            v = j / tube_segments
            phi = j * 2.0 * math.pi / tube_segments
            cos_ph = math.cos(phi)
            sin_ph = math.sin(phi)

            x = (radius_main + radius_tube * cos_ph) * cos_th
            y = radius_tube * sin_ph
            z = (radius_main + radius_tube * cos_ph) * sin_th

            nx = cos_ph * cos_th
            ny = sin_ph
            nz = cos_ph * sin_th

            data.extend([x, y, z, nx, ny, nz, u, v])

    for i in range(main_segments):
        for j in range(tube_segments):
            k1 = i * (tube_segments + 1) + j
            k2 = k1 + tube_segments + 1

            indices.extend([k1, k1 + 1, k2])
            indices.extend([k1 + 1, k2 + 1, k2])

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)