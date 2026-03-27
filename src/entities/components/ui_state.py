from dataclasses import dataclass, field
from enum import Enum

from meshes.mesh import Mesh
from shading.material import Material


class AddType(Enum):
    Plane = 0
    Cube = 1
    UVSphere = 2
    Tetrahedron = 3
    PointLight = 4
    Camera = 5


@dataclass
class UiState:
    default_material: Material

    selected_entity: int | None = None

    add_mesh_type: AddType = AddType.Cube
    sphere_radius: float = 0.5
    uv_sphere_stacks: int = 20
    uv_sphere_sectors: int = 40
    general_mesh_size: float = 1.0
    preview_visual_initialized: bool = False

    should_close_add_menu: bool = False

    entities_to_dispose: list[int] = field(default_factory=lambda: [])