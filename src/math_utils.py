import math
import numpy as np
import numpy.typing as npt


def float1(x):
    return np.float32(x)


def vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


def vec4(x, y, z, w):
    return np.array([x, y, z, w], dtype=np.float32)


def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def create_perspective_projection(fovy, aspect, near, far):
    tan_half_fovy = math.tan(math.radians(fovy) / 2.0)

    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = 1.0 / (aspect * tan_half_fovy)
    result[1, 1] = 1.0 / tan_half_fovy
    result[2, 2] = -(far + near) / (far - near)
    result[2, 3] = -(2.0 * far * near) / (far - near)
    result[3, 2] = -1.0

    return result


def create_orthographic_projection(left, right, bottom, top, near, far):
    result = np.identity(4, dtype=np.float32)
    result[0, 0] = 2.0 / (right - left)
    result[1, 1] = 2.0 / (top - bottom)
    result[2, 2] = -2.0 / (far - near)
    result[0, 3] = -(right + left) / (right - left)
    result[1, 3] = -(top + bottom) / (top - bottom)
    result[2, 3] = -(far + near) / (far - near)
    return result


def create_look_at(eye, target, up):
    z_axis = normalize(eye - target)
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    result = np.identity(4, dtype=np.float32)

    result[0, :3] = x_axis
    result[1, :3] = y_axis
    result[2, :3] = z_axis

    result[0, 3] = -np.dot(x_axis, eye)
    result[1, 3] = -np.dot(y_axis, eye)
    result[2, 3] = -np.dot(z_axis, eye)

    return result


def calculate_direction_from_rotation(rotation_degrees):
    rad = np.radians(rotation_degrees)
    cx, cy, cz = np.cos(rad)
    sx, sy, sz = np.sin(rad)

    mat_rot_x = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype=np.float32)

    mat_rot_y = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype=np.float32)

    mat_rot_z = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    mat_rot = mat_rot_z @ mat_rot_y @ mat_rot_x

    default_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return mat_rot @ default_dir


def create_transformation_matrix(position, rotation_euler, scale) -> npt.NDArray[np.float32]:
    """
    Creates a model matrix (translation * rotation * scale).
    rotation_euler: (x_degrees, y_degrees, z_degrees)
    """
    mat_trans = np.identity(4, dtype=np.float32)
    mat_trans[0, 3] = position[0]
    mat_trans[1, 3] = position[1]
    mat_trans[2, 3] = position[2]

    rx, ry, rz = np.radians(rotation_euler)

    c, s = np.cos(rz), np.sin(rz)
    mat_rot_z = np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    c, s = np.cos(ry), np.sin(ry)
    mat_rot_y = np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    c, s = np.cos(rx), np.sin(rx)
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

    return mat_trans @ mat_rot @ mat_scale  # type: ignore
