from dataclasses import dataclass


@dataclass(slots=True, eq=False)
class SceneAnimatorState:
    animate_vehicles: bool = False
