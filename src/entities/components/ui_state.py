from dataclasses import dataclass, field
from enum import Enum, auto

from meshes.mesh import Mesh
from shading.material import Material


class AddType(Enum):
    Plane = auto()
    Cube = auto()
    Tetrahedron = auto()
    Prism = auto()
    Cone = auto()
    Cylinder = auto()
    UVSphere = auto()
    Tetrasphere = auto()
    Icosphere = auto()
    Torus = auto()
    PointLight = auto()
    Camera = auto()


@dataclass
class UiState:
    default_material: Material

    # == mesh creation configuration state ==

    add_mesh_type: AddType = AddType.Cube
    
    # spheres
    sphere_radius: float = 0.5
    uv_sphere_stacks: int = 20
    uv_sphere_sectors: int = 40
    subdiv_sphere_subdivisions: int = 3

    # cylinders, cones & prisms
    column_radius_bottom: float = 1.0
    cylinder_radius_top: float = 1.0
    column_height: float = 2.0
    column_sectors: int = 32

    # tori
    torus_main_radius: float = 1.0
    torus_tube_radius: float = 1.0
    torus_main_sectors: int = 32
    torus_tube_sectors: int = 16

    # cubes & tetrahedrons
    general_mesh_size: float = 1.0

    preview_visual_initialized: bool = False

    # == other stuff ==
    selected_entity: int | None = None
    should_close_add_menu: bool = False
    entities_to_dispose: list[int] = field(default_factory=lambda: [])