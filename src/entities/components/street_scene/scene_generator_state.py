from typing import Any
from dataclasses import dataclass, field

from entities.components.visuals.assets import Mesh, ModelAsset
from entities.components.visuals.material import Material


@dataclass(slots=True, eq=False)
class SceneGeneratorState:
    should_generate: bool = False

    cube_mesh: Any = None
    plane_mesh: Any = None

    car_model: ModelAsset | None = None
    bus_model: ModelAsset | None = None

    street_length: float = 300.0
    street_width: float = 10.0
    sidewalk_width: float = 5.0

    building_count: int = 30
    min_building_height: float = 5.0
    max_building_height: float = 20.0
    min_building_width: float = 4.0
    max_building_width: float = 8.0

    vehicle_count: int = 30
    min_vehicle_speed: float = 5.0
    max_vehicle_speed: float = 20.0

    lanes_per_direction: int = 2
