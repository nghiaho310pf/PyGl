import OpenGL.GL as GL


class UniformBuffer:
    def __init__(self, size, binding_point=0):
        self.id = GL.glGenBuffers(1)
        self.size = size
        self.binding_point = binding_point

        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.id)
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, self.size, None, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

        GL.glBindBufferRange(GL.GL_UNIFORM_BUFFER, self.binding_point, self.id, 0, self.size)

    def update(self, data: bytes, offset=0):
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.id)
        GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, offset, len(data), data)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

    def destroy(self):
        GL.glDeleteBuffers(1, [self.id])