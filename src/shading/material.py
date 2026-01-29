from shading.shader import Shader


class Material:
    def __init__(self, shader: Shader, properties: dict = None):
        self.shader = shader
        self.properties = properties if properties else {}

    def set_property(self, name, value):
        self.properties[name] = value

    def use(self):
        self.shader.use()

        for name, value in self.properties.items():
            if isinstance(value, float):
                self.shader.set_float(name, value)
            elif isinstance(value, int):
                self.shader.set_int(name, value)
            elif isinstance(value, (list, tuple)):
                if len(value) == 3:
                    self.shader.set_vec3(name, value)
                elif len(value) == 4:
                    self.shader.set_vec4(name, value)
                    pass
