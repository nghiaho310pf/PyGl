import numpy as np
import math

from entities.components.rotated_camera import RotatedCamera
from entities.components.transform import Transform
from entities.components.camera import Camera
from entities.registry import Registry

class CameraRotatorSystem:
    def __init__(self, registry: Registry):
        self.registry = registry

    def update(self, time: float, delta_time: float):
        for entity, (transform, camera, rotated_camera) in self.registry.view(Transform, Camera, RotatedCamera):
            transform.rotation += delta_time * rotated_camera.rotation_delta
            transform.rotation %= 360.0

            rad = np.radians(transform.rotation)
            front = np.array([
                math.cos(rad[1]) * math.cos(rad[0]),
                math.sin(rad[0]),
                math.sin(rad[1]) * math.cos(rad[0])
            ])

            transform.position = rotated_camera.center - (front * rotated_camera.distance)