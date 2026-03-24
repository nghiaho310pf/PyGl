from entities.components.rotated import Rotated
from entities.components.transform import Transform
from entities.registry import Registry


class RotatorSystem:
    def update(self, registry: Registry, time: float, delta_time: float):
        for entity, (transform, rotated) in registry.view(Transform, Rotated):
            transform.rotation += delta_time * rotated.delta
            transform.rotation %= 360.0
