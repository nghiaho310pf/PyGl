from dataclasses import dataclass, field
from enum import Enum

from geometry.mesh import Mesh
from shading.material import Material


class AddType(Enum):
    Sphere = 0
    Cube = 1
    Plane = 2
    PointLight = 3
    Camera = 4


@dataclass
class UiState:
    default_material: Material

    selected_entity: int | None = None

    add_mesh_type: AddType = AddType.Cube
    sphere_radius: float = 0.5
    sphere_stacks: int = 20
    sphere_sectors: int = 40
    cube_size: float = 1.0
    preview_visual_initialized: bool = False

    should_close_add_menu: bool = False

    entities_to_dispose: list[int] = field(default_factory=lambda: [])