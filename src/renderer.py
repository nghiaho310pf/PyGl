import OpenGL.GL as GL
import numpy as np

from application import Application
from mesh import Mesh
from shader import Shader

VS_SOURCE = """
#version 330 core
layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_UV;

void main() {
    gl_Position = vec4(a_Position, 1.0);
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

        vertices = np.array([
            -0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0, # bottom left
             0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  1.0,  0.0, # bottom right
             0.0,  0.5,  0.0,  0.0,  0.0,  1.0,  0.5,  1.0  # top center
        ], dtype=np.float32)

        self.triangle_mesh = Mesh(vertices)

    def render(self, window_size: tuple[int, int]):
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.shader.use()
        self.triangle_mesh.draw()
