from entities.registry import Registry
from entities.components.transform import Transform
import math_utils


class TransformInheritanceSystem:
    @staticmethod
    def update(registry: Registry):
        updated_entities: set[int] = set()
        transform_entities = registry.get_components_of_type(Transform).keys()

        for entity in transform_entities:
            if entity not in updated_entities:
                TransformInheritanceSystem._update_entity(registry, entity, updated_entities)

    @staticmethod
    def _update_entity(registry: Registry, entity: int, updated_entities: set[int]):
        parent = registry.get_parent(entity)

        if parent is not None and parent not in updated_entities:
            if parent in registry.get_components_of_type(Transform):
                TransformInheritanceSystem._update_entity(registry, parent, updated_entities)

        r_transform = registry.get_components(entity, Transform)
        if not r_transform:
            updated_entities.add(entity)
            return

        transform = r_transform[0]
        local = transform.local
        world = transform.world

        parent_transform = None
        if parent is not None:
            r_parent_transform = registry.get_components(parent, Transform)
            if r_parent_transform:
                parent_transform = r_parent_transform[0]

        if transform.inherit and parent_transform is not None:
            parent_world = parent_transform.world

            world.rotation[:] = math_utils.quaternion_mul(parent_world.rotation, local.rotation)
            world.scale[:] = parent_world.scale * local.scale

            rotated_local_pos = math_utils.rotate_vector_by_quaternion(
                local.position * parent_world.scale,
                parent_world.rotation
            )
            world.position[:] = parent_world.position + rotated_local_pos
        else:
            world.position[:] = local.position
            world.rotation[:] = local.rotation
            world.scale[:] = local.scale

        updated_entities.add(entity)

        for child in registry.get_children(entity):
            if child not in updated_entities:
                TransformInheritanceSystem._update_entity(registry, child, updated_entities)
