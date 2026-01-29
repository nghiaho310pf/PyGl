import math

import numpy as np

import math_utils
from entities.components.transform import Transform
from entities.components.rotated import Rotated
from entities.registry import Registry


class RotatorSystem:
    def __init__(self, registry: Registry):
        self.registry = registry

    def update(self, time: float, delta_time: float):
        for entity, (transform, rotated) in self.registry.view(Transform, Rotated):
            transform.rotation += delta_time * rotated.delta
