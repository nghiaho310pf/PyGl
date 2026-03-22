import warnings

from shading.shader import Shader


class Material:
    def __init__(self, shader: Shader, properties: dict | None = None):
        self.shader = shader
        self.properties = properties if properties else {}

    def set_property(self, name, value):
        self.properties[name] = value

    def setup_properties(self):
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

    def use(self):
        warnings.warn(
            "Material#use is deprecated. Batch draw calls by shader and use Shader#use() + Material#setup_properties instead",
            category=DeprecationWarning,
            stacklevel=2
        )
        self.shader.use()
        self.setup_properties()
