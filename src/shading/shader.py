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
