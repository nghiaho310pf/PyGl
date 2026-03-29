from dataclasses import dataclass, field


@dataclass
class Disposal:
    entities_to_dispose: set[int] = field(default_factory=set)
