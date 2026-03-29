from dataclasses import dataclass, field
from enum import Enum, auto

from meshes.mesh import Mesh
from shading.material import Material


class AddType(Enum):
    Triangle = auto()
    Plane = auto()
    Pentagon = auto()
    Hexagon = auto()
    Circle = auto()
    Ellipse = auto()
    Trapezoid = auto()
    Star = auto()
    Arrow = auto()

    FunctionSurface = auto()

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
    preview_entity: int
    default_material: Material

    # == mesh creation configuration state ==

    add_mesh_type: AddType = AddType.Cube

    # circles & ellipses
    ellipse_radius_x: float = 1.2
    ellipse_radius_z: float = 0.8
    round_surface_sides: int = 32

    # trapezoids
    trapezoid_top_width: float = 0.8
    trapezoid_bottom_width: float = 1.2
    trapezoid_height: float = 1.0

    # stars
    star_outer_radius: float = 0.4
    star_inner_radius: float = 1.2
    star_points: int = 4

    # arrows
    arrow_tail_length: float = 1.0
 
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

    general_mesh_size: float = 1.0

    preview_visual_initialized: bool = False

    # == other stuff ==
    selected_entity: int | None = None
    should_close_add_menu: bool = False
    entities_to_dispose: list[int] = field(default_factory=lambda: [])