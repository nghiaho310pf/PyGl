import struct
import numpy as np

import OpenGL.GL as GL


class Shader:
    def __init__(self, vertex_source: str, fragment_source: str):
        self.program = self._compile_program(vertex_source, fragment_source)
        self.uniform_cache = {}

    def use(self):
        GL.glUseProgram(self.program)

    @staticmethod
    def _compile_program(vert_src, frag_src):
        def compile_src(src, shader_type):
            shader = GL.glCreateShader(shader_type)
            GL.glShaderSource(shader, src)
            GL.glCompileShader(shader)

            if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
                err = GL.glGetShaderInfoLog(shader).decode()
                shader_type_str = "Vertex" if shader_type == GL.GL_VERTEX_SHADER else "Fragment"
                raise RuntimeError(f"{shader_type_str} shader compilation error:\n{err}")
            return shader

        vs = compile_src(vert_src, GL.GL_VERTEX_SHADER)
        fs = compile_src(frag_src, GL.GL_FRAGMENT_SHADER)

        program = GL.glCreateProgram()
        GL.glAttachShader(program, vs)
        GL.glAttachShader(program, fs)
        GL.glLinkProgram(program)

        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            error = GL.glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"Shader linking error:\n{error}")

        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

        return program

    def _get_uniform_location(self, name):
        if name in self.uniform_cache:
            return self.uniform_cache[name]

        loc = GL.glGetUniformLocation(self.program, name)
        # note: loc will be -1 if the uniform is not found or optimized out by the GPU
        self.uniform_cache[name] = loc
        return loc

    def set_vec3(self, name, value):
        loc = self._get_uniform_location(name)
        if loc != -1:
            GL.glUniform3fv(loc, 1, value)

    def set_vec4(self, name, value):
        loc = self._get_uniform_location(name)
        if loc != -1:
            GL.glUniform4fv(loc, 1, value)

    def set_float(self, name, value):
        loc = self._get_uniform_location(name)
        if loc != -1:
            GL.glUniform1f(loc, value)

    def set_int(self, name, value):
        loc = self._get_uniform_location(name)
        if loc != -1:
            GL.glUniform1i(loc, value)

    def set_mat4(self, name, matrix):
        loc = GL.glGetUniformLocation(self.program, name)
        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, matrix)

    def set_vec3_array(self, name, values):
        loc = self._get_uniform_location(name)
        if loc != -1:
            flattened_values = [coord for vec in values for coord in vec]
            GL.glUniform3fv(loc, len(values), flattened_values)

    def set_mat4_array(self, name, matrices):
        loc = self._get_uniform_location(name)
        if loc != -1:
            flattened_matrices = np.array(matrices, dtype=np.float32).flatten()
            GL.glUniformMatrix4fv(loc, len(matrices), GL.GL_FALSE, flattened_matrices)

    def set_int_array(self, name, values):
        loc = self._get_uniform_location(name)
        if loc != -1:
            GL.glUniform1iv(loc, len(values), np.array(values, dtype=np.int32))


class ShaderGlobals:
    """
        Manages the 'SceneData' UBO block shared by shaders.
        std140 layout:
        - mat4  u_Projection (64 bytes)
        - mat4  u_View       (64 bytes)
        - vec3  u_ViewPos    (12 bytes)
        - float u_Time       (4 bytes)
    """
    BINDING_POINT = 0

    def __init__(self):
        self.size = 144
        self.buffer_id = GL.glGenBuffers(1)

        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.buffer_id)
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, self.size, None, GL.GL_DYNAMIC_DRAW)
        GL.glBindBufferRange(GL.GL_UNIFORM_BUFFER, self.BINDING_POINT, self.buffer_id, 0, self.size)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

    def update(self, projection_matrix, view_matrix, camera_position, time: float):
        data = (
                projection_matrix.tobytes() +
                view_matrix.tobytes() +
                struct.pack('3ff', camera_position[0], camera_position[1], camera_position[2], time)
        )

        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.buffer_id)
        GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, 0, len(data), data)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

    def attach_to(self, shader: Shader):
        """
        Tells a specific shader to look at Binding Point 0 for "SceneData"
        """
        block_index = GL.glGetUniformBlockIndex(shader.program, "SceneData")
        if block_index != GL.GL_INVALID_INDEX:
            GL.glUniformBlockBinding(shader.program, block_index, self.BINDING_POINT)
