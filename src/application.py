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
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)

        self.win = glfw.create_window(width, height, "PyGl", None, None)

        glfw.make_context_current(self.win)

        glfw.set_framebuffer_size_callback(self.win, self._on_resize_internal)

        version: bytes = GL.glGetString(GL.GL_VERSION)
        glsl_version: bytes = GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)
        renderer: bytes = GL.glGetString(GL.GL_RENDERER)
        print(f"OpenGL: {version.decode()}\nGLSL: {glsl_version.decode()}\nRenderer: {renderer.decode()}")

        # Enable Vsync
        glfw.swap_interval(1)

    def _on_resize_internal(self, window, width, height):
        if width == 0 or height == 0:  # may happen when window is minimized
            return

        GL.glViewport(0, 0, width, height)
        self.on_resize()

    def get_window_size(self):
        return glfw.get_window_size(self.win)

    def get_time(self):
        return glfw.get_time()

    # == Orchestration ==

    def run(self):
        while not glfw.window_should_close(self.win):
            self.render()

            glfw.swap_buffers(self.win)
            glfw.poll_events()

        glfw.terminate()

    # == Overrideable callbacks ==

    def on_resize(self):
        pass

    def render(self):
        pass
