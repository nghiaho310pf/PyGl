import OpenGL.GL as GL
import numpy as np

import blinn_phong
from application import Application
from camera import Camera
from entity import Entity
from geometry import generate_sphere
from material import Material
from mesh import Mesh


class RotatingEntity(Entity):
    def update(self, dt: float):
        self.rotation[1] += 1.5


class Renderer(Application):
    def __init__(self, width, height):
        super().__init__(width, height)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_MULTISAMPLE)

        self.camera = Camera(position=[0.0, 0.0, 2.5], aspect_ratio=width / height)

        shader = blinn_phong.make_shader()

        mat_orange = Material(shader, {
            "u_Color": [1.0, 0.5, 0.2]
        })
        mat_blue = Material(shader, {
            "u_Color": [0.459, 0.651, 1.0]
        })

        # vertices = np.array([
        #     -0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0, # bottom left
        #      0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  1.0,  0.0, # bottom right
        #      0.0,  0.5,  0.0,  0.0,  0.0,  1.0,  0.5,  1.0  # top center
        # ], dtype=np.float32)
        sphere_vertices, sphere_indices = generate_sphere(radius=0.5, stacks=32, sectors=32)
        self.mesh = Mesh(sphere_vertices, sphere_indices)

        self.entities = [
            RotatingEntity(self.mesh, mat_orange, position=(-0.6, 0, 0)),
            RotatingEntity(self.mesh, mat_blue, position=(0.6, 0, 0))
        ]

        self.light_pos = [2.0, 2.0, 5.0]

        self.last_update = None

    def on_resize(self):
        pass

    def render(self):
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        window_width, window_height = self.get_window_size()
        self.camera.aspect_ratio = window_width / window_height

        proj = self.camera.get_projection_matrix()
        view = self.camera.get_view_matrix()

        now = self.get_time()
        if self.last_update is None:
            self.last_update = now
        dt = now - self.last_update

        for entity in self.entities:
            entity.update(dt)

            entity.material.use()

            shader = entity.material.shader
            shader.set_mat4("u_Projection", proj)
            shader.set_mat4("u_View", view)
            shader.set_vec3("u_LightPos", self.light_pos)
            shader.set_vec3("u_ViewPos", self.camera.position)

            model = entity.get_model_matrix()
            shader.set_mat4("u_Model", model)

            entity.mesh.draw()

        self.last_update = now
