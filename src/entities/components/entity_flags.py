from dataclasses import dataclass


@dataclass(slots=True, eq=False)
class EntityFlags:
    name: str | None = None
    classification: int = 0
    is_internal: bool = False
    dispose_alongside_parent: bool = True
