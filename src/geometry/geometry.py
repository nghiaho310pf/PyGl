import math

import numpy as np


def generate_sphere(radius=1.0, stacks=16, sectors=32):
    """
    Generates a UV sphere.

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3)]
        indices: np.array (uint32) -> EBO indices
    """
    data = []
    indices = []

    for i in range(stacks + 1):
        stack_angle = math.pi / 2 - i * math.pi / stacks
        xy = radius * math.cos(stack_angle)
        y = radius * math.sin(stack_angle)

        for j in range(sectors + 1):
            sector_angle = j * 2 * math.pi / sectors

            x = xy * math.sin(sector_angle)
            z = xy * math.cos(sector_angle)

            nx = x / radius
            ny = y / radius
            nz = z / radius

            data.extend([x, y, z, nx, ny, nz])

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
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3)]
        indices: np.array (uint32) -> EBO indices
    """
    s = size / 2.0

    front = [
        -s, s, s, 0, 0, 1,
        -s, -s, s, 0, 0, 1,
        s, -s, s, 0, 0, 1,
        s, s, s, 0, 0, 1,
    ]

    back = [
        s, s, -s, 0, 0, -1,
        s, -s, -s, 0, 0, -1,
        -s, -s, -s, 0, 0, -1,
        -s, s, -s, 0, 0, -1,
    ]

    top = [
        -s, s, -s, 0, 1, 0,
        -s, s, s, 0, 1, 0,
        s, s, s, 0, 1, 0,
        s, s, -s, 0, 1, 0,
    ]

    bottom = [
        -s, -s, s, 0, -1, 0,
        -s, -s, -s, 0, -1, 0,
        s, -s, -s, 0, -1, 0,
        s, -s, s, 0, -1, 0,
    ]

    right = [
        s, s, s, 1, 0, 0,
        s, -s, s, 1, 0, 0,
        s, -s, -s, 1, 0, 0,
        s, s, -s, 1, 0, 0,
    ]

    left = [
        -s, s, -s, -1, 0, 0,
        -s, -s, -s, -1, 0, 0,
        -s, -s, s, -1, 0, 0,
        -s, s, s, -1, 0, 0,
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


def generate_plane(size=10.0):
    """
    Generates a flat plane (floor) on the XZ plane.
    Total vertices: 4
    Total indices: 6 (2 triangles)

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3)]
        indices: np.array (uint32) -> EBO indices
    """
    s = size / 2.0

    data = [
        -s, 0.0, -s, 0, 1, 0,
        -s, 0.0, s, 0, 1, 0,
        s, 0.0, s, 0, 1, 0,
        s, 0.0, -s, 0, 1, 0,
    ]

    indices = [
        0, 1, 2,
        0, 2, 3
    ]

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)

def generate_cylinder(radius=1.0, height=2.0, sectors=32):
    """
    Generates a UV cylinder with a single stack (two rings for the barrel),
    plus flat circular caps on both ends.

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3)]
        indices: np.array (uint32) -> EBO indices
    """
    data = []
    indices = []

    h = height / 2.0

    # Bottom ring
    for j in range(sectors + 1):
        u = j / sectors
        angle = u * 2 * math.pi
        nx = math.cos(angle)
        nz = math.sin(angle)
        data.extend([radius * nx, -h, radius * nz, nx, 0.0, nz])

    # Top ring
    for j in range(sectors + 1):
        u = j / sectors
        angle = u * 2 * math.pi
        nx = math.cos(angle)
        nz = math.sin(angle)
        data.extend([radius * nx, h, radius * nz, nx, 0.0, nz])

    # Barrel indices
    for j in range(sectors):
        # From bottom ring
        k1 = j
        # From top ring
        k2 = j + sectors + 1

        indices.extend([k1, k1 + 1, k2])
        indices.extend([k1 + 1, k2 + 1, k2])

    # Top face
    top_center_idx = len(data) // 6
    data.extend([0.0, h, 0.0, 0.0, 1.0, 0.0])

    for j in range(sectors + 1):
        u_angle = j / sectors
        angle = u_angle * 2 * math.pi
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)

        data.extend([x, h, z, 0.0, 1.0, 0.0])

    for j in range(sectors):
        indices.extend([top_center_idx, top_center_idx + 1 + j, top_center_idx + 2 + j])

    # Bottom face
    bottom_center_idx = len(data) // 6
    data.extend([0.0, -h, 0.0, 0.0, -1.0, 0.0])

    for j in range(sectors + 1):
        u_angle = j / sectors
        angle = u_angle * 2 * math.pi
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)

        data.extend([x, -h, z, 0.0, -1.0, 0.0])

    for j in range(sectors):
        indices.extend([bottom_center_idx, bottom_center_idx + 2 + j, bottom_center_idx + 1 + j])

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)