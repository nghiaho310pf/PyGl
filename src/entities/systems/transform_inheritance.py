from entities.registry import Registry
from entities.components.transform import Transform
import math_utils
import numpy as np


class TransformInheritanceSystem:
    @staticmethod
    def update(registry: Registry):
        transforms_dict = registry.get_components_of_type(Transform)
        if not transforms_dict:
            return

        scaled_pos_scratch = np.zeros(3, dtype=np.float32)

        stack: list[int] = []
        for entity in transforms_dict:
            parent = registry.get_parent(entity)
            if parent is None or parent not in transforms_dict:
                stack.append(entity)

        while stack:
            entity = stack.pop()
            transform = transforms_dict[entity]

            parent = registry.get_parent(entity)
            parent_transform = transforms_dict.get(parent) if parent is not None else None

            local = transform.local
            world = transform.world

            if transform.inherit and parent_transform is not None:
                parent_world = parent_transform.world

                math_utils.quaternion_mul_out(parent_world.rotation, local.rotation, world.rotation)
                np.multiply(parent_world.scale, local.scale, out=world.scale)

                np.multiply(local.position, parent_world.scale, out=scaled_pos_scratch)
                math_utils.rotate_vector_by_quaternion_out(
                    scaled_pos_scratch,
                    parent_world.rotation,
                    world.position
                )

                world.position += parent_world.position
            else:
                world.position[:] = local.position
                world.rotation[:] = local.rotation
                world.scale[:] = local.scale

            for child in registry.get_children(entity):
                if child in transforms_dict:
                    stack.append(child)
