import OpenGL.GL as GL
import numpy as np
from application import Application
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
            -0.5, -0.5, 0.0,
            0.5, -0.5, 0.0,
            0.0, 0.5, 0.0
        ], dtype=np.float32)

        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 3 * 4, None)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

    def render(self, window_size: tuple[int, int]):
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.shader.use()
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)