import math
from typing import Tuple
import numpy as np
import numpy.typing as npt


def float1(x):
    return np.float32(x)


def vec2(x, y):
    return np.array([x, y], dtype=np.float32)


def unpack_vec2(v: npt.NDArray) -> Tuple[float, float]:
    return (float(v[0]), float(v[1]))


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
    q = quaternion_from_euler(rotation_degrees)
    default_dir = vec3(0.0, 0.0, -1.0)
    return rotate_vector_by_quaternion(default_dir, q)


def quaternion_from_euler(euler_degrees):
    rad = np.radians(euler_degrees.astype(np.float64)) * 0.5
    sx, sy, sz = np.sin(rad)
    cx, cy, cz = np.cos(rad)

    return vec4(
        sx * cy * cz - cx * sy * sz, # x
        cx * sy * cz + sx * cy * sz, # y
        cx * cy * sz - sx * sy * cz, # z
        cx * cy * cz + sx * sy * sz  # w
    )


def quaternion_to_euler(q: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    q64 = q.astype(np.float64)
    x, y, z, w = q64

    sinp = 2 * (w * y - z * x)

    if abs(sinp) >= 1.0 - 1e-7:
        pitch = np.copysign(np.pi / 2, sinp)
        roll = 2 * np.atan2(x, w)
        yaw = 0.0
    else:
        pitch = np.asin(sinp)

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.atan2(sinr_cosp, cosr_cosp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.atan2(siny_cosp, cosy_cosp)

    return np.degrees(vec3(roll, pitch, yaw)).astype(np.float32)


def quaternions_from_euler(euler_degrees):
    rad = np.radians(euler_degrees.astype(np.float64)) * 0.5
    sx, sy, sz = np.sin(rad)
    cx, cy, cz = np.cos(rad)

    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    w = cx * cy * cz + sx * sy * sz

    q1 = (x, y, z, w)
    q2 = (-x, -y, -z, -w)

    sorted_q1, sorted_q2 = sorted([q1, q2])

    return vec4(*sorted_q1), vec4(*sorted_q2)


def minimize_euler(degrees: npt.NDArray[np.float32]):
    sol1 = (degrees + 180.0) % 360.0 - 180.0

    alt_degrees = np.array([degrees[0] + 180.0, 180.0 - degrees[1], degrees[2] + 180.0])
    sol2 = (alt_degrees + 180.0) % 360.0 - 180.0

    dist1_sq = np.sum(sol1 ** 2)
    dist2_sq = np.sum(sol2 ** 2)

    return sol1 if dist1_sq < dist2_sq else sol2


def quaternion_to_axes(q):
    mat = quaternion_matrix(q)
    right = mat[:3, 0]
    up = mat[:3, 1]
    front = mat[:3, 2]
    return right, up, front


def rotate_vector_by_quaternion(v, q):
    q_vec = q[:3]
    q_w = q[3]
    t = 2.0 * np.cross(q_vec, v)
    return v + q_w * t + np.cross(q_vec, t)


def quaternion_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return vec4(
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    )


def quaternion_matrix(q):
    q = normalize(q)
    x, y, z, w = q

    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    return np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy),       0.0],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx),       0.0],
        [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx + yy), 0.0],
        [0.0,                 0.0,                 0.0,                 1.0]
    ], dtype=np.float32)


def quaternion_slerp(q0, q1, fraction):
    q0 = normalize(q0)
    q1 = normalize(q1)

    dot = np.dot(q0, q1)

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        return normalize(q0 + fraction * (q1 - q0))

    theta_0 = math.acos(dot)
    theta = theta_0 * fraction

    q2 = normalize(q1 - q0 * dot)

    return q0 * math.cos(theta) + q2 * math.sin(theta)


def quaternion_identity():
    return vec4(0.0, 0.0, 0.0, 1.0)


def quaternion_from_axis_angle(axis, angle_degrees):
    axis = normalize(axis)
    half_angle = math.radians(angle_degrees) * 0.5
    s = math.sin(half_angle)
    return vec4(
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
        math.cos(half_angle)
    )


def create_transformation_matrix(position, rotation_quaternion, scale) -> npt.NDArray[np.float32]:
    """
    Creates a model matrix (translation * rotation * scale).
    rotation_quaternion: (x, y, z, w)
    """
    mat = quaternion_matrix(rotation_quaternion)

    mat[:3, 0] *= scale[0]
    mat[:3, 1] *= scale[1]
    mat[:3, 2] *= scale[2]

    mat[:3, 3] = position
    
    return mat
