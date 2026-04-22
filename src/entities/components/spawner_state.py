from dataclasses import dataclass, field

from entities.components.transform import Transform
from entities.components.visuals.assets import ModelAsset


@dataclass(slots=True, eq=False)
class SpawnRequest:
    model: ModelAsset
    root_transform: Transform
    parent_entity: int | None = None


@dataclass(slots=True, eq=False)
class SpawnerState:
    pending_spawns: list[SpawnRequest] = field(default_factory=list)
