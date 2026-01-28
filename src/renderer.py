import OpenGL.GL as GL

from application import Application

class Renderer(Application):
    def __init__(self, width, height):
        super().__init__(width, height)

    def render(self, window_size: tuple[int, int]):
        GL.glClearColor(0.5, 0.5, 0.5, 0.1)
        #GL.glEnable(GL.GL_CULL_FACE) # enable backface culling (Exercise 1)
        #GL.glFrontFace(GL.GL_CCW) # GL_CCW: default

        GL.glEnable(GL.GL_DEPTH_TEST) # enable depth test (Exercise 1)
        GL.glDepthFunc(GL.GL_LESS) # GL_LESS: default

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
