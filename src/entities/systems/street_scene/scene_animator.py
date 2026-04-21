from entities.registry import Registry
from entities.components.transform import Transform
from entities.components.street_scene.scene_animator_state import SceneAnimatorState
from entities.components.street_scene.scene_generator_state import SceneGeneratorState
from entities.components.street_scene.vehicle import Vehicle


class SceneAnimatorSystem:
    @staticmethod
    def update(registry: Registry, dt: float):
        for scene_entity, (anim_state, gen_state) in registry.view(SceneAnimatorState, SceneGeneratorState):
            if not anim_state.animate_vehicles:
                continue

            street_length = gen_state.street_length
            half_length = street_length / 2.0

            children = registry.get_children(scene_entity)
            for child in children:
                r_vehicle = registry.get_components(child, Transform, Vehicle)
                if r_vehicle:
                    transform, vehicle = r_vehicle
                    if vehicle.direction is None:
                        continue

                    transform.local.position += vehicle.direction * vehicle.speed * dt

                    if transform.local.position[2] > half_length:
                        transform.local.position[2] = -half_length
                    elif transform.local.position[2] < -half_length:
                        transform.local.position[2] = half_length
