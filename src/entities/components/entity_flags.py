from dataclasses import dataclass
from enum import IntEnum


class EntityClassification(IntEnum):
    Environment = 0
    Vehicle = 1
    Building = 2
    Human = 3


@dataclass(slots=True, eq=False)
class EntityFlags:
    name: str | None = None
    classification: EntityClassification = EntityClassification.Environment
    is_internal: bool = False
    dispose_alongside_parent: bool = True
