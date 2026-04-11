from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List
import numpy as np
import numpy.typing as npt
from math_utils import vec3 


class OptimizerAlgorithm(Enum):
    BatchGD = auto()
    SGD = auto()
    MiniBatchSGD = auto()
    Momentum = auto()


@dataclass(slots=True)
class OptimizerState:
    algorithm: OptimizerAlgorithm = OptimizerAlgorithm.BatchGD
    
    # hyperparameters
    learning_rate: float = 0.001
    momentum_rate: float = 0.9
    noise_scale: float = 5.0

    # internal state tracking
    velocity_x: float = 0.0
    velocity_z: float = 0.0

    # history for rendering the line
    trajectory: List[npt.NDArray] = field(default_factory=list)
