import math

import numpy as np
from numba import njit, float32

vec = float32[::1]
mat = float32[:, ::1]


@njit(vec(float32, float32, float32), cache=True, parallel=False)
def vec3(x: float, y: float, z: float):
    return np.array([x, y, z])


@njit(vec(vec), cache=True, parallel=False)
def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


@njit(mat(float32, float32, float32, float32, mat), cache=True, parallel=False)
def compute_projection_matrix(fovy, aspect, near, far, out):
    tan_half_fovy = np.tan(np.radians(fovy) / 2.0)

    out[0, 0] = 1.0 / (aspect * tan_half_fovy)
    out[1, 1] = 1.0 / tan_half_fovy
    out[2, 2] = -(far + near) / (far - near)
    out[2, 3] = -1.0
    out[3, 2] = -(2.0 * far * near) / (far - near)
    out[3, 3] = 0.0

    return out


@njit(mat(vec, vec, vec, mat), cache=True, parallel=False)
def compute_look_at_matrix(eye, target, up, out=None):
    z_axis = normalize(eye - target)  # Forward (inverted for GL)
    x_axis = normalize(np.cross(up, z_axis))  # Right
    y_axis = np.cross(z_axis, x_axis)  # Up re-calculated

    if out is None:
        out = np.identity(4, dtype=np.float32)

    out[0, 0] = x_axis[0]
    out[1, 0] = x_axis[1]
    out[2, 0] = x_axis[2]

    out[0, 1] = y_axis[0]
    out[1, 1] = y_axis[1]
    out[2, 1] = y_axis[2]

    out[0, 2] = z_axis[0]
    out[1, 2] = z_axis[1]
    out[2, 2] = z_axis[2]

    out[3, 0] = -np.dot(x_axis, eye)
    out[3, 1] = -np.dot(y_axis, eye)
    out[3, 2] = -np.dot(z_axis, eye)

    return out


@njit(mat(vec, vec, vec, mat), cache=True, parallel=False)
def compute_transformation_matrix(position, rotation_euler, scale, out):
    rx = math.radians(rotation_euler[0])
    ry = math.radians(rotation_euler[1])
    rz = math.radians(rotation_euler[2])

    cx, sx_sin = math.cos(rx), math.sin(rx)
    cy, sy_sin = math.cos(ry), math.sin(ry)
    cz, sz_sin = math.cos(rz), math.sin(rz)

    # Rotation + Scale components
    # Row 0
    out[0, 0] = (cz * cy) * scale[0]
    out[0, 1] = (cz * sy_sin * sx_sin - sz_sin * cx) * scale[0]
    out[0, 2] = (cz * sy_sin * cx + sz_sin * sx_sin) * scale[0]
    out[0, 3] = 0.0

    # Row 1
    out[1, 0] = (sz_sin * cy) * scale[1]
    out[1, 1] = (sz_sin * sy_sin * sx_sin + cz * cx) * scale[1]
    out[1, 2] = (sz_sin * sy_sin * cx - cz * sx_sin) * scale[1]
    out[1, 3] = 0.0

    # Row 2
    out[2, 0] = (-sy_sin) * scale[2]
    out[2, 1] = (cy * sx_sin) * scale[2]
    out[2, 2] = (cy * cx) * scale[2]
    out[2, 3] = 0.0

    # Row 3 (Translation)
    out[3, 0] = position[0]
    out[3, 1] = position[1]
    out[3, 2] = position[2]
    out[3, 3] = 1.0

    return out
