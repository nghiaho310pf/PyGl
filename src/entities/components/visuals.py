from dataclasses import dataclass

from geometry.mesh import Mesh
from shading.material import Material


@dataclass
class Visuals:
    mesh: Mesh
    material: Material
