from dataclasses import dataclass, field


@dataclass(slots=True, eq=False)
class Disposal:
    entities_to_dispose: set[int] = field(default_factory=set)
