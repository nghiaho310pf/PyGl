import OpenGL.GL as GL
import numpy as np

from application import Application
from camera import Camera
from entity import Entity
from material import Material
from mesh import Mesh
from shader import Shader

VS_SOURCE = """
#version 330 core
layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_UV;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;

void main() {
    gl_Position = u_Projection * u_View * u_Model * vec4(a_Position, 1.0);
}
"""

FS_SOURCE = """
#version 330 core
out vec4 FragColor;
uniform vec4 u_Color;

void main() {
    FragColor = u_Color;
}
"""


class Renderer(Application):
    def __init__(self, width, height):
        super().__init__(width, height)

        base_shader = Shader(VS_SOURCE, FS_SOURCE)

        mat_orange = Material(base_shader, [1.0, 0.5, 0.2, 1.0])
        mat_blue = Material(base_shader, [0.459, 0.651, 1.0, 1.0])

        self.camera = Camera(position=[0.0, 0.0, 3.0], aspect_ratio=width / height)

        vertices = np.array([
            -0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0, # bottom left
             0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  1.0,  0.0, # bottom right
             0.0,  0.5,  0.0,  0.0,  0.0,  1.0,  0.5,  1.0  # top center
        ], dtype=np.float32)
        self.mesh = Mesh(vertices)

        self.entities = [
            Entity(self.mesh, mat_orange, position=(-0.6, 0, 0)),
            Entity(self.mesh, mat_blue, position=(0.6, 0, 0))
        ]

    def on_resize(self):
        pass

    def render(self):
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        window_width, window_height = self.get_window_size()
        self.camera.aspect_ratio = window_width / window_height

        proj = self.camera.get_projection_matrix()
        view = self.camera.get_view_matrix()

        for entity in self.entities:
            entity.material.use()

            shader = entity.material.shader
            shader.set_mat4("u_Projection", proj)
            shader.set_mat4("u_View", view)

            model = entity.get_model_matrix()
            shader.set_mat4("u_Model", model)

            entity.mesh.draw()

            entity.rotation[1] += 1.5
