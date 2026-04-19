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
    x, y, z, w = rotation_quaternion

    mag2 = x*x + y*y + z*z + w*w
    if abs(mag2 - 1.0) > 1e-6 and mag2 > 1e-8:
        mag = mag2 ** 0.5
        x /= mag; y /= mag; z /= mag; w /= mag

    x2, y2, z2 = x * 2.0, y * 2.0, z * 2.0
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2

    sx, sy, sz = scale
    px, py, pz = position

    return np.array((
        (1.0 - (yy + zz)) * sx, (xy - wz) * sy,         (xz + wy) * sz,         px,
        (xy + wz) * sx,         (1.0 - (xx + zz)) * sy, (yz - wx) * sz,         py,
        (xz - wy) * sx,         (yz + wx) * sy,         (1.0 - (xx + yy)) * sz, pz,
        0.0,                    0.0,                    0.0,                    1.0
    ), dtype=np.float32).reshape((4, 4))


def get_frustum_corners_world_space(proj_matrix: npt.NDArray[np.float32], view_matrix: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
    inv_vp = np.linalg.inv(proj_matrix @ view_matrix)
    corners = []
    for x in [-1.0, 1.0]:
        for y in [-1.0, 1.0]:
            for z in [-1.0, 1.0]:
                pt = inv_vp @ np.array([x, y, z, 1.0], dtype=np.float32)
                corners.append(pt[:3] / pt[3])
    return corners


def get_light_space_matrix(proj_matrix: npt.NDArray[np.float32], view_matrix: npt.NDArray[np.float32], light_dir: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    corners = get_frustum_corners_world_space(proj_matrix, view_matrix)

    center = np.mean(corners, axis=0)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(np.dot(normalize(light_dir), up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    light_view = create_look_at(center - light_dir * 50.0, center, up)

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')

    for corner in corners:
        trf = light_view @ np.append(corner, 1.0)
        trf = trf[:3]
        min_x = min(min_x, trf[0])
        max_x = max(max_x, trf[0])
        min_y = min(min_y, trf[1])
        max_y = max(max_y, trf[1])
        min_z = min(min_z, trf[2])
        max_z = max(max_z, trf[2])

    # small margin to prevent PCF clipping at the edges
    width = max_x - min_x
    height = max_y - min_y
    margin_scale = 0.02
    min_x -= width * margin_scale
    max_x += width * margin_scale
    min_y -= height * margin_scale
    max_y += height * margin_scale

    z_extension = 100.0
    near_plane = -max_z - z_extension
    far_plane = -min_z + z_extension

    light_proj = create_orthographic_projection(min_x, max_x, min_y, max_y, near_plane, far_plane)

    return light_proj @ light_view
