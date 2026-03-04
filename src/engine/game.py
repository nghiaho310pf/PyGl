import numpy as np

from engine.application import Application
from entities.components.camera import Camera
from entities.components.point_light import PointLight
from entities.components.rotated import Rotated
from entities.components.rotated_camera import RotatedCamera
from entities.components.transform import Transform
from entities.components.visuals import Visuals
from entities.registry import Registry
from entities.systems.camera_rotator import CameraRotatorSystem
from entities.systems.render import RenderSystem
from entities.systems.rotator import RotatorSystem
from geometry.geometry import generate_plane, generate_cylinder
from geometry.mesh import Mesh
from math_utils import vec3
from shading.material import Material
from shading.shaders import blinn_phong


class Game(Application):
    def __init__(self, initial_window_width, initial_window_height):
        super().__init__(initial_window_width, initial_window_height)

        self.registry = Registry()

        self.render_system = RenderSystem(self.registry)
        self.rotator_system = RotatorSystem(self.registry)
        self.camera_rotator_system = CameraRotatorSystem(self.registry)

        # == demo setup ==

        shader = blinn_phong.make_shader()
        self.render_system.attach_shader(shader)

        mat_orange = Material(shader, {
            "u_Albedo": [1.0, 0.318, 0.133],
        })
        mat_blue = Material(shader, {
            "u_Albedo": [0.276, 0.481, 1.0],
        })
        mat_grey = Material(shader, {
            "u_Albedo": [0.08, 0.08, 0.08],
        })

        plane_vertices, plane_indices = generate_plane()
        cylinder_vertices, cylinder_indices = generate_cylinder(0.5, 2, 32)

        # orange cylinder entity
        e1 = self.registry.create_entity()
        self.registry.add_components(
            e1,
            Transform(position=vec3(0.0, 0.5, 0)),
            Visuals(Mesh(cylinder_vertices, cylinder_indices), mat_orange)
        )

        # floor entity
        e3 = self.registry.create_entity()
        self.registry.add_components(
            e3,
            Transform(position=vec3(0.0, 0.0, 0.0)),
            Visuals(Mesh(plane_vertices, plane_indices), mat_grey)
        )

        # camera entities
        c1 = self.registry.create_entity()
        self.registry.add_components(
            c1,
            Transform(rotation=vec3(-20.0, 0.0, 0.0)),
            Camera(),
            RotatedCamera(center=vec3(0.0, 1.0, 0.0), distance=np.float32(6.0), rotation_delta=vec3(0.0, 45.0, 0.0))
        )

        c2 = self.registry.create_entity()
        self.registry.add_components(
            c2,
            Transform(
                position=vec3(0.0, 2.4, 5.0),
                rotation=vec3(-15.0, -90.0, 0.0)
            ),
            Camera()
        )

        # point light entity
        l = self.registry.create_entity()
        self.registry.add_components(
            l,
            Transform(position=vec3(1.2, 4.0, 1.2)),
            PointLight(color=vec3(220.0, 220.0, 220.0))
        )

        self.last_update = None

    def render(self):
        now = self.get_time()
        if self.last_update is None:
            self.last_update = now
        dt = now - self.last_update

        self.rotator_system.update(now, dt)
        self.camera_rotator_system.update(now, dt)
        self.render_system.update(self.get_window_size(), now, dt)

        self.last_update = now
