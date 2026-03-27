import warnings
from enum import Enum, auto
from dataclasses import dataclass, field

from shading.shader import Shader


class ShaderType(Enum):
    Flat = 0
    BlinnPhong = 1
    Gouraud = 2


@dataclass(eq=False)
class Material:
    shader_type: ShaderType
    properties: dict = field(default_factory=dict)
