from dataclasses import dataclass

from geometry.mesh import Mesh
from shading.material import Material


@dataclass
class UiState:
    selected_entity: int | None = None
