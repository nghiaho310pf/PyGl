from shader import Shader


class Material:
    def __init__(self, shader: Shader, color: list[float]):
        self.shader = shader
        self.color = color

    def use(self):
        self.shader.use()
        try:
            self.shader.set_vec4("u_Color", self.color)
        except:
            pass
