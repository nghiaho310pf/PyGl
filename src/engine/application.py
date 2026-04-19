import ctypes
import platform
import time

import OpenGL.GL as GL
import glfw
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from imgui_bundle import imgui, hello_imgui


class Application:
    win: ctypes.c_void_p
    imgui_renderer: GlfwRenderer

    current_window_width: int
    current_window_height: int

    has_broken_opengl = platform.system() == "Darwin"

    def __init__(self, initial_window_width, initial_window_height):
        self.current_window_width = initial_window_width
        self.current_window_height = initial_window_height

        glfw.init()

        glfw.default_window_hints()

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1 if Application.has_broken_opengl else 5)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        glfw.window_hint(glfw.SRGB_CAPABLE, True)

        glfw.window_hint(glfw.RED_BITS, 8)
        glfw.window_hint(glfw.GREEN_BITS, 8)
        glfw.window_hint(glfw.BLUE_BITS, 8)
        glfw.window_hint(glfw.DEPTH_BITS, 24)

        self.win = glfw.create_window(initial_window_width, initial_window_height, "PyGl", None, None)

        glfw.make_context_current(self.win)

        GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)
        glfw.set_framebuffer_size_callback(self.win, self._on_resize_internal)

        imgui.create_context()
        self.imgui_renderer = GlfwRenderer(self.win)
        font_config = imgui.ImFontConfig()
        imgui.get_io().fonts.add_font_from_file_ttf(
            hello_imgui.asset_file_full_path("fonts/DroidSans.ttf"),
            16,
            font_config
        )
        font_config.merge_mode = True
        imgui.get_io().fonts.add_font_from_file_ttf(
            hello_imgui.asset_file_full_path("fonts/Font_Awesome_6_Free-Solid-900.otf"),
            16,
            font_config
        )

        version: bytes = GL.glGetString(GL.GL_VERSION)  # type: ignore
        glsl_version: bytes = GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)  # type: ignore
        renderer: bytes = GL.glGetString(GL.GL_RENDERER)  # type: ignore
        print(f"OpenGL: {version.decode()}\nGLSL: {glsl_version.decode()}\nRenderer: {renderer.decode()}")

        # disable vsync. we'll try to render at a fixed frame rate in Application#run
        glfw.swap_interval(0)

    def _on_resize_internal(self, window, width, height):
        if width == 0 or height == 0:  # may happen when window is minimized
            return

        self.current_window_width = width
        self.current_window_height = height

        GL.glViewport(0, 0, width, height)
        self.on_resize()

    def get_window_size(self):
        width, height = glfw.get_window_size(self.win)
        if width == 0 or height == 0:  # may happen when window is minimized
            width = self.current_window_width
            height = self.current_window_height
        return (width, height)

    def get_time(self):
        return glfw.get_time()

    # == Orchestration ==

    def run(self):
        target_fps = 120
        frame_time_target = 1.0 / target_fps

        print(f"Targeting {target_fps} FPS")

        last_time = glfw.get_time()

        while not glfw.window_should_close(self.win):
            current_time = glfw.get_time()
            delta = current_time - last_time

            if delta >= frame_time_target:
                last_time = current_time

                self.render()

                glfw.swap_buffers(self.win)
                glfw.poll_events()
            else:
                time.sleep(0.0001)

        self.imgui_renderer.shutdown()
        imgui.destroy_context()
        glfw.terminate()

    # == Overrideable callbacks ==

    def on_resize(self):
        pass

    def render(self):
        pass
