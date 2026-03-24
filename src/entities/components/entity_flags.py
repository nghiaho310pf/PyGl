from dataclasses import dataclass


@dataclass
class EntityFlags:
    name: str | None = None
    is_internal: bool = False
