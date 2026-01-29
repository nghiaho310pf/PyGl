import math

import numpy as np

import math_utils


class Camera:
    def __init__(self, position, fov=45.0, aspect_ratio=16 / 9, near=0.1, far=100.0):
        self.position = np.array(position, dtype=np.float32)

        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.yaw = -90.0
        self.pitch = 0.0

        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far

        self.update_camera_vectors()

    def get_view_matrix(self):
        return math_utils.create_look_at(self.position, self.position + self.front, self.up)

    def get_projection_matrix(self):
        return math_utils.create_perspective_projection(self.fov, self.aspect_ratio, self.near, self.far)

    def update_camera_vectors(self):
        front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ])
        self.front = math_utils.normalize(front)

        self.right = math_utils.normalize(np.cross(self.front, self.world_up))
        self.up = math_utils.normalize(np.cross(self.right, self.front))
