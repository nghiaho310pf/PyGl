import OpenGL.GL as GL
import numpy as np

from application import Application
from camera import Camera
from entity import Entity
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

void main() {
    FragColor = vec4(0.459, 0.651, 1, 1.0); // Engineer blue
}
"""


class Renderer(Application):
    def __init__(self, width, height):
        super().__init__(width, height)

        self.shader = Shader(VS_SOURCE, FS_SOURCE)

        self.camera = Camera(position=[0.0, 0.0, 3.0], aspect_ratio=width / height)

        vertices = np.array([
            -0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0, # bottom left
             0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  1.0,  0.0, # bottom right
             0.0,  0.5,  0.0,  0.0,  0.0,  1.0,  0.5,  1.0  # top center
        ], dtype=np.float32)
        self.mesh = Mesh(vertices)

        self.triangle_entity = Entity(
            mesh=self.mesh,
            position=(0, 0, 0)
        )

    def render(self, window_size: tuple[int, int]):
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.shader.use()

        proj = self.camera.get_projection_matrix()
        view = self.camera.get_view_matrix()

        self.shader.set_mat4("u_Projection", proj)
        self.shader.set_mat4("u_View", view)

        self.triangle_entity.rotation[1] += 1.5

        model = self.triangle_entity.get_model_matrix()
        self.shader.set_mat4("u_Model", model)

        self.triangle_entity.mesh.draw()
