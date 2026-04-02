from dataclasses import dataclass, field


@dataclass(slots=True, eq=False)
class UiState:
    selection_child_entity: int
