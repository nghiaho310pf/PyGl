import OpenGL.GL as GL
import glfw

class Application:
    win: any

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

        print("OpenGL", GL.glGetString(GL.GL_VERSION).decode() + ", GLSL",
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ", Renderer", GL.glGetString(GL.GL_RENDERER).decode())

    def run(self):
        while not glfw.window_should_close(self.win):
            self.render(glfw.get_window_size(self.win))

            glfw.swap_buffers(self.win)
            glfw.poll_events()
        
        glfw.terminate()
    
    def render(self, window_size: tuple[int, int]):
        pass
