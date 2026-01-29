import math

import numpy as np


def vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def create_perspective_projection(fovy, aspect, near, far):
    tan_half_fovy = math.tan(math.radians(fovy) / 2.0)

    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = 1.0 / (aspect * tan_half_fovy)
    result[1, 1] = 1.0 / tan_half_fovy
    result[2, 2] = -(far + near) / (far - near)
    result[2, 3] = -1.0
    result[3, 2] = -(2.0 * far * near) / (far - near)

    return result


def create_look_at(eye, target, up):
    z_axis = normalize(eye - target)  # Forward (inverted for GL)
    x_axis = normalize(np.cross(up, z_axis))  # Right
    y_axis = np.cross(z_axis, x_axis)  # Up re-calculated

    result = np.identity(4, dtype=np.float32)
    result[0, 0] = x_axis[0]
    result[1, 0] = x_axis[1]
    result[2, 0] = x_axis[2]

    result[0, 1] = y_axis[0]
    result[1, 1] = y_axis[1]
    result[2, 1] = y_axis[2]

    result[0, 2] = z_axis[0]
    result[1, 2] = z_axis[1]
    result[2, 2] = z_axis[2]

    result[3, 0] = -np.dot(x_axis, eye)
    result[3, 1] = -np.dot(y_axis, eye)
    result[3, 2] = -np.dot(z_axis, eye)

    return result


def create_transformation_matrix(position, rotation_euler, scale):
    """
    Creates a model matrix (translation * rotation * scale).
    rotation_euler: (x_degrees, y_degrees, z_degrees)
    """
    mat_trans = np.identity(4, dtype=np.float32)
    mat_trans[3, 0] = position[0]
    mat_trans[3, 1] = position[1]
    mat_trans[3, 2] = position[2]

    rx, ry, rz = np.radians(rotation_euler)

    c, s = math.cos(rz), math.sin(rz)
    mat_rot_z = np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    c, s = math.cos(ry), math.sin(ry)
    mat_rot_y = np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    c, s = math.cos(rx), math.sin(rx)
    mat_rot_x = np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    mat_rot = mat_rot_z @ mat_rot_y @ mat_rot_x

    mat_scale = np.identity(4, dtype=np.float32)
    mat_scale[0, 0] = scale[0]
    mat_scale[1, 1] = scale[1]
    mat_scale[2, 2] = scale[2]

    return mat_trans @ mat_rot @ mat_scale
