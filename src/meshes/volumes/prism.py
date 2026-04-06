# src/meshes/geometry/prism.py
import math
import numpy as np


def generate_prism(sides=6, radius=1.0, height=2.0):
    """
    Generates a flat-shaded prism.

    Returns:
        vertices: np.array (float32) -> Interleaved [Pos(3), Norm(3), UV(2)]
        indices: np.array (uint32) -> EBO indices
    """
    data = []
    indices = []

    for i in range(sides):
        theta0 = i * 2.0 * math.pi / sides
        theta1 = (i + 1) * 2.0 * math.pi / sides

        theta_mid = (i + 0.5) * 2.0 * math.pi / sides
        nx = math.cos(theta_mid)
        ny = 0.0
        nz = math.sin(theta_mid)

        x0, z0 = radius * math.cos(theta0), radius * math.sin(theta0)
        x1, z1 = radius * math.cos(theta1), radius * math.sin(theta1)

        u0 = i / sides
        u1 = (i + 1) / sides

        v_offset = len(data) // 8

        data.extend([x0, -height / 2.0, z0, nx, ny, nz, u0, 0.0])  # bottom right
        data.extend([x0, height / 2.0, z0, nx, ny, nz, u0, 1.0])  # top right
        data.extend([x1, height / 2.0, z1, nx, ny, nz, u1, 1.0])  # top left
        data.extend([x1, -height / 2.0, z1, nx, ny, nz, u1, 0.0])  # bottom left

        indices.extend([v_offset, v_offset + 1, v_offset + 2])
        indices.extend([v_offset, v_offset + 2, v_offset + 3])

    top_offset = len(data) // 8
    data.extend([0.0, height / 2.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
    for i in range(sides + 1):
        theta = i * 2.0 * math.pi / sides
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        data.extend([
            x, height / 2.0, z,
            0.0, 1.0, 0.0,
            math.cos(theta)*0.5+0.5, math.sin(theta)*0.5+0.5
        ])
    for i in range(sides):
        indices.extend([top_offset, top_offset + i + 2, top_offset + i + 1])

    bottom_offset = len(data) // 8
    data.extend([0.0, -height / 2.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.5])
    for i in range(sides + 1):
        theta = i * 2.0 * math.pi / sides
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        data.extend([
            x, -height / 2.0, z,
            0.0, -1.0, 0.0,
            math.cos(theta)*0.5+0.5, math.sin(theta)*0.5+0.5
        ])
    for i in range(sides):
        indices.extend([bottom_offset, bottom_offset +
                       i + 1, bottom_offset + i + 2])

    return np.array(data, dtype=np.float32), np.array(indices, dtype=np.uint32)
