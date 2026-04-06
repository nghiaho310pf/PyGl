from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import numpy.typing as npt


class CompilationStatus(Enum):
    Ok = auto()
    Warning = auto()
    Error = auto()


@dataclass
class SurfaceFunction:
    expression: str = "0"
    expression_dirty: bool = False
    generated: bool = False

    size: float = 10.0
    resolution: int = 50

    error_status: CompilationStatus = CompilationStatus.Ok
    error_string: str = ""
