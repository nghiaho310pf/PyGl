import numpy as np
import math


def generate_sphere(radius=1.0, stacks=16, sectors=32):
    """
    Generates a UV sphere.
    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    data = []
    indices = []

    for i in range(stacks + 1):
        stack_angle = math.pi / 2 - i * math.pi / stacks
        xy = radius * math.cos(stack_angle)
        z = radius * math.sin(stack_angle)

        for j in range(sectors + 1):
            sector_angle = j * 2 * math.pi / sectors

            x = xy * math.cos(sector_angle)
            y = xy * math.sin(sector_angle)

            nx = x / radius
            ny = y / radius
            nz = z / radius

            u = j / sectors
            v = i / stacks

            data.extend([x, y, z, nx, ny, nz, u, v])

    for i in range(stacks):
        k1 = i * (sectors + 1)
        k2 = k1 + sectors + 1

        for j in range(sectors):
            if i != 0:
                indices.extend([k1, k2, k1 + 1])

            if i != (stacks - 1):
                indices.extend([k1 + 1, k2, k2 + 1])

            k1 += 1
            k2 += 1

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)