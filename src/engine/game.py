import numpy as np

from engine.application import Application
from entities.components.camera import Camera
from entities.components.point_light import PointLight
from entities.components.rotated import Rotated
from entities.components.transform import Transform
from entities.components.visuals import Visuals
from entities.registry import Registry
from entities.systems.render import RenderSystem
from entities.systems.rotator import RotatorSystem
from geometry.geometry import generate_sphere, generate_cube_flat, generate_plane
from geometry.mesh import Mesh
from math_utils import vec3
from shading.material import Material
from shading.shaders import tf2_ggx_smith


class Game(Application):
    def __init__(self, initial_window_width, initial_window_height):
        super().__init__(initial_window_width, initial_window_height)

        self.registry = Registry()

        self.render_system = RenderSystem(self.registry)
        self.rotator_system = RotatorSystem(self.registry)

        # == demo setup ==

        shader = tf2_ggx_smith.make_shader()
        self.render_system.attach_shader(shader)

        mat_orange = Material(shader, {
            "u_Albedo": [1.0, 0.5, 0.2],
            "u_Metallic": 0.3,
            "u_Roughness": 0.3,
            "u_Reflectance": 0.25,
            "u_Translucency": 0.25,
            "u_AO": 0.1,
        })
        mat_blue = Material(shader, {
            "u_Albedo": [0.459, 0.651, 1.0],
            "u_Metallic": 0.7,
            "u_Roughness": 0.3,
            "u_Reflectance": 0.0,
            "u_Translucency": 0.0,
            "u_AO": 0.1,
        })
        mat_grey = Material(shader, {
            "u_Albedo": [0.225, 0.225, 0.225],
            "u_Metallic": 0.7,
            "u_Roughness": 0.3,
            "u_Reflectance": 0.0,
            "u_Translucency": 0.0,
            "u_AO": 0.1,
        })

        sphere_vertices, sphere_indices = generate_sphere(radius=0.5, stacks=48, sectors=48)
        cube_vertices, cube_indices = generate_cube_flat(size=1.0)
        plane_vertices, plane_indices = generate_plane()

        # orange sphere entity
        e1 = self.registry.create_entity()
        self.registry.add_component(e1, Transform(position=vec3(0.0, 0.5, 0)))
        self.registry.add_component(e1, Visuals(Mesh(sphere_vertices, sphere_indices), mat_orange))

        # blue cube entity
        e2 = self.registry.create_entity()
        self.registry.add_component(e2, Transform(position=vec3(0.0, 1.5, 0)))
        self.registry.add_component(e2, Visuals(Mesh(cube_vertices, cube_indices), mat_blue))
        self.registry.add_component(e2, Rotated(delta=vec3(0.0, 90.0, 0.0)))

        # floor entity
        e3 = self.registry.create_entity()
        self.registry.add_component(e3, Transform(position=vec3(0.0, 0.0, 0.0)))
        self.registry.add_component(e3, Visuals(Mesh(plane_vertices, plane_indices), mat_grey))

        # camera entity
        c = self.registry.create_entity()
        self.registry.add_component(c, Transform(position=vec3(0.0, 2.4, 5.0),
                                                 rotation=vec3(-20.0, -100.0, 0.0)))
        self.registry.add_component(c, Camera())

        # point light entities
        c = self.registry.create_entity()
        self.registry.add_component(c, Transform(position=vec3(1.2, 4.0, 1.2)))
        self.registry.add_component(c, PointLight(color=vec3(300.0, 300.0, 300.0)))

        c = self.registry.create_entity()
        self.registry.add_component(c, Transform(position=vec3(-1.0, 5.0, -1.0)))
        self.registry.add_component(c, PointLight(color=vec3(100.0, 100.0, 100.0), radius=0.1))

        self.last_update = None

    def render(self):
        now = self.get_time()
        if self.last_update is None:
            self.last_update = now
        dt = now - self.last_update

        self.rotator_system.update(now, dt)
        self.render_system.update(self.get_window_size(), now, dt)

        self.last_update = now
