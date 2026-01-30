import math

import numpy as np


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


def generate_cube_flat(size=1.0):
    """
    Generates a flat shaded cube.
    Total vertices: 24 = 4 per face * 6 faces
    Total indices: 36 = 6 per face * 6 faces

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    s = size / 2.0

    front = [
        -s, s, s, 0, 0, 1, 0, 1,
        -s, -s, s, 0, 0, 1, 0, 0,
        s, -s, s, 0, 0, 1, 1, 0,
        s, s, s, 0, 0, 1, 1, 1
    ]

    back = [
        s, s, -s, 0, 0, -1, 0, 1,
        s, -s, -s, 0, 0, -1, 0, 0,
        -s, -s, -s, 0, 0, -1, 1, 0,
        -s, s, -s, 0, 0, -1, 1, 1
    ]

    top = [
        -s, s, -s, 0, 1, 0, 0, 1,
        -s, s, s, 0, 1, 0, 0, 0,
        s, s, s, 0, 1, 0, 1, 0,
        s, s, -s, 0, 1, 0, 1, 1
    ]

    bottom = [
        -s, -s, s, 0, -1, 0, 0, 1,
        -s, -s, -s, 0, -1, 0, 0, 0,
        s, -s, -s, 0, -1, 0, 1, 0,
        s, -s, s, 0, -1, 0, 1, 1
    ]

    right = [
        s, s, s, 1, 0, 0, 0, 1,
        s, -s, s, 1, 0, 0, 0, 0,
        s, -s, -s, 1, 0, 0, 1, 0,
        s, s, -s, 1, 0, 0, 1, 1
    ]

    left = [
        -s, s, -s, -1, 0, 0, 0, 1,
        -s, -s, -s, -1, 0, 0, 0, 0,
        -s, -s, s, -1, 0, 0, 1, 0,
        -s, s, s, -1, 0, 0, 1, 1
    ]

    vertices = np.array(front + back + top + bottom + right + left, dtype=np.float32)

    indices = []

    for i in range(6):
        offset = i * 4
        indices.extend([
            offset + 0, offset + 1, offset + 2,
            offset + 0, offset + 2, offset + 3
        ])

    return vertices, np.array(indices, dtype=np.uint32)


def generate_plane(size=10.0, uv_scale=1.0):
    """
    Generates a flat plane (floor) on the XZ plane.
    Total vertices: 4
    Total indices: 6 (2 triangles)

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    s = size / 2.0

    data = [
        -s, 0.0, -s, 0, 1, 0, 0, uv_scale,
        -s, 0.0, s, 0, 1, 0, 0, 0,
        s, 0.0, s, 0, 1, 0, uv_scale, 0,
        s, 0.0, -s, 0, 1, 0, uv_scale, uv_scale
    ]

    indices = [
        0, 1, 2,
        0, 2, 3
    ]

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)