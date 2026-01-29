import ctypes

import OpenGL.GL as GL


class Mesh:
    def __init__(self, vertices, indices=None):
        """
        :param vertices: numpy array (float32) containing interleaved data [Px, Py, Pz, Nx, Ny, Nz, U, V ...]
        :param indices: numpy array (uint32) containing triangle indices (optional)
        """
        self.vertices = vertices
        self.indices = indices

        self.vao = None
        self.vbo = None
        self.ebo = None

        # pretty much fixed stride of 8 (xyz; nxnynz; uv)
        self.stride = 8

        self.vertex_count = len(vertices) // self.stride
        self.indices_count = len(indices) if indices is not None else 0

        self._setup_mesh()

    def _setup_mesh(self):
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL.GL_STATIC_DRAW)

        if self.indices is not None:
            self.ebo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL.GL_STATIC_DRAW)

        stride = self.stride * 4

        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))

        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))

        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(24))

        GL.glBindVertexArray(0)

    def draw(self):
        GL.glBindVertexArray(self.vao)

        if self.indices is not None:
            GL.glDrawElements(GL.GL_TRIANGLES, self.indices_count, GL.GL_UNSIGNED_INT, None)
        else:
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)

        GL.glBindVertexArray(0)

    def destroy(self):
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(1, [self.vbo])
        if self.ebo:
            GL.glDeleteBuffers(1, [self.ebo])
