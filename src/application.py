import ctypes

import OpenGL.GL as GL
import glfw

class Application:
    win: ctypes.c_void_p

    def __init__(self, width, height):
        glfw.init()

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)

        self.win = glfw.create_window(width, height, "PyGl", None, None)

        glfw.make_context_current(self.win)

        version: bytes = GL.glGetString(GL.GL_VERSION)
        glsl_version: bytes = GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)
        renderer: bytes = GL.glGetString(GL.GL_RENDERER)
        print(f"OpenGL: {version.decode()}\nGLSL: {glsl_version.decode()}\nRenderer: {renderer.decode()}")

        # Enable Vsync
        glfw.swap_interval(1)

    def run(self):
        while not glfw.window_should_close(self.win):
            self.render(glfw.get_window_size(self.win))

            glfw.swap_buffers(self.win)
            glfw.poll_events()
        
        glfw.terminate()
    
    def render(self, window_size: tuple[int, int]):
        pass
