from dataclasses import dataclass

from meshes.mesh import Mesh
from shading.material import Material


@dataclass
class Visuals:
    mesh: Mesh
    material: Material
    enabled: bool = True
    is_internal: bool = False
