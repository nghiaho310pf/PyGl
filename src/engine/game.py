import numpy as np

from engine.application import Application
from entities.components.camera import Camera
from entities.components.rotated import Rotated
from entities.components.transform import Transform
from entities.components.visuals import Visuals
from entities.registry import Registry
from entities.systems.render import RenderSystem
from entities.systems.rotator import RotatorSystem
from geometry.geometry import generate_sphere, generate_cube_flat
from geometry.mesh import Mesh
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

        # vertices = np.array([
        #     -0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0, # bottom left
        #      0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  1.0,  0.0, # bottom right
        #      0.0,  0.5,  0.0,  0.0,  0.0,  1.0,  0.5,  1.0  # top center
        # ], dtype=np.float32)
        sphere_vertices, sphere_indices = generate_sphere(radius=0.5, stacks=48, sectors=48)
        cube_vertices, cube_indices = generate_cube_flat(size=1.0)

        # orange sphere entity
        e1 = self.registry.create_entity()
        self.registry.add_component(e1, Transform(position=np.array([-0.7, 0, 0])))
        self.registry.add_component(e1, Visuals(Mesh(sphere_vertices, sphere_indices), mat_orange))

        # blue cube entity
        e2 = self.registry.create_entity()
        self.registry.add_component(e2, Transform(position=np.array([0.7, 0, 0])))
        self.registry.add_component(e2, Visuals(Mesh(cube_vertices, cube_indices), mat_blue))
        self.registry.add_component(e2, Rotated(delta=np.array([0.0, 90.0, 0.0])))

        # camera entity
        c = self.registry.create_entity()
        self.registry.add_component(c, Transform(position=np.array([0.0, 1.0, 2.5]),
                                                 rotation=np.array([-22.0, -90.0, 0.0])))
        self.registry.add_component(c, Camera())

        self.last_update = None

    def render(self):
        now = self.get_time()
        if self.last_update is None:
            self.last_update = now
        dt = now - self.last_update

        self.rotator_system.update(now, dt)
        self.render_system.update(self.get_window_size(), now, dt)

        self.last_update = now
