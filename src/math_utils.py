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

    default_dir = vec3(0.0, 0.0, -1.0)
    return mat_rot @ default_dir


def quaternion_from_euler(euler_degrees):
    rad = np.radians(euler_degrees) * 0.5
    sx, sy, sz = np.sin(rad)
    cx, cy, cz = np.cos(rad)

    return vec4(
        sx * cy * cz - cx * sy * sz, # x
        cx * sy * cz + sx * cy * sz, # y
        cx * cy * sz - sx * sy * cz, # z
        cx * cy * cz + sx * sy * sz  # w
    )


def quaternion_to_euler(q):
    x, y, z, w = q

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # suppress negative zeroes
    return np.degrees(vec3(roll, pitch, yaw)) + float1(0.0)


def quaternions_from_euler(euler_degrees):
    rad = np.radians(euler_degrees) * 0.5
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


def fit_euler(new_degrees: npt.NDArray[np.float32], old_degrees: npt.NDArray[np.float32]):
    alt_degrees = np.array([new_degrees[0] + 180.0, 180.0 - new_degrees[1], new_degrees[2] + 180.0])

    sol1 = new_degrees + np.round((old_degrees - new_degrees) / 360.0) * 360.0
    sol2 = alt_degrees + np.round((old_degrees - alt_degrees) / 360.0) * 360.0

    dist1_sq = np.sum((sol1 - old_degrees) ** 2)
    dist2_sq = np.sum((sol2 - old_degrees) ** 2)

    return sol1 if dist1_sq < dist2_sq else sol2


def quaternion_to_axes(q):
    mat = quaternion_matrix(q)
    right = mat[:3, 0]
    up = mat[:3, 1]
    front = mat[:3, 2]
    return right, up, front


def rotate_vector_by_quaternion(v, q):
    v_q = vec4(v[0], v[1], v[2], 0.0)
    q_conj = vec4(-q[0], -q[1], -q[2], q[3])
    return quaternion_mul(quaternion_mul(q, v_q), q_conj)[:3]


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


def create_transformation_matrix(position, rotation_quaternion, scale) -> npt.NDArray[np.float32]:
    """
    Creates a model matrix (translation * rotation * scale).
    rotation_quaternion: (x, y, z, w)
    """
    mat_trans = np.identity(4, dtype=np.float32)
    mat_trans[0, 3] = position[0]
    mat_trans[1, 3] = position[1]
    mat_trans[2, 3] = position[2]

    mat_rot = quaternion_matrix(rotation_quaternion)

    mat_scale = np.identity(4, dtype=np.float32)
    mat_scale[0, 0] = scale[0]
    mat_scale[1, 1] = scale[1]
    mat_scale[2, 2] = scale[2]

    return mat_trans @ mat_rot @ mat_scale  # type: ignore
