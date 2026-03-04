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
from geometry.geometry import generate_sphere, generate_cube_flat, generate_plane
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

        sphere_vertices, sphere_indices = generate_sphere(radius=0.5, stacks=20, sectors=40)
        cube_vertices, cube_indices = generate_cube_flat(size=1.0)
        plane_vertices, plane_indices = generate_plane()

        # orange sphere entity
        e1 = self.registry.create_entity()
        self.registry.add_components(
            e1,
            Transform(position=vec3(0.0, 0.5, 0)),
            Visuals(Mesh(sphere_vertices, sphere_indices), mat_orange)
        )

        # blue cube entity
        e2 = self.registry.create_entity()
        self.registry.add_components(
            e2,
            Transform(position=vec3(0.0, 1.5, 0)),
            Visuals(Mesh(cube_vertices, cube_indices), mat_blue)
        )

        # floor entity
        e3 = self.registry.create_entity()
        self.registry.add_components(
            e3,
            Transform(position=vec3(0.0, 0.0, 0.0)),
            Visuals(Mesh(plane_vertices, plane_indices), mat_grey)
        )

        # camera entity
        c = self.registry.create_entity()
        self.registry.add_components(
            c,
            Transform(rotation=vec3(-20.0, 0.0, 0.0)),
            Camera(),
            RotatedCamera(center=vec3(0.0, 1.0, 0.0), distance=np.float32(6.0), rotation_delta=vec3(0.0, 45.0, 0.0))
        )

        # point light entity
        c = self.registry.create_entity()
        self.registry.add_components(
            c,
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
        self.render_system.update(self.get_window_size(), now, dt)
        self.camera_rotator_system.update(now, dt)

        self.last_update = now
