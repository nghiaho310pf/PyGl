from dataclasses import dataclass
from enum import Enum, auto


class LossFunctionType(Enum):
    Himmelblau = auto()
    Rosenbrock = auto()
    Booth = auto()


@dataclass
class GradientDescentSurface:
    function_type: LossFunctionType = LossFunctionType.Himmelblau
    dirty: bool = True

    size: float = 10.0
    resolution: int = 100

    # specific parameters for Rosenbrock
    rosenbrock_a: float = 1.0
    rosenbrock_b: float = 100.0

    # running state
    is_running: bool = False
    step_timer: float = 0.0
    step_interval: float = 0.016667
    iterations: int = 0
