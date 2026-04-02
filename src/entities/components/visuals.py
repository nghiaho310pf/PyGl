from dataclasses import dataclass
from enum import Enum

from meshes.mesh import Mesh
from shading.material import Material


class DrawMode(Enum):
    Normal = 0
    Wireframe = 1


@dataclass(slots=True, eq=False)
class Visuals:
    mesh: Mesh
    material: Material
    enabled: bool = True
    draw_mode: DrawMode = DrawMode.Normal

    # == non-editable properties ==
    cull_back_faces: bool = True
    is_internal: bool = False
