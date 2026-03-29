from dataclasses import dataclass


@dataclass
class Camera:
    fov: float = 45.0
    near: float = 0.1
    far: float = 100.0

    focal_point_distance = 5.0
