import random

from entities.registry import Registry
from entities.components.transform import Transform
from entities.components.visuals.visuals import Visuals
from entities.components.visuals.assets import AssetsState
from entities.components.visuals.material import Material
from entities.components.entity_flags import EntityClassification, EntityFlags
from entities.components.disposal import Disposal
from entities.components.street_scene.scene_generator_state import SceneGeneratorState
from entities.components.street_scene.vehicle import Vehicle
from entities.systems.assets import AssetSystem
from meshes.volumes.cube import generate_cube
from meshes.surfaces.plane import generate_plane
from math_utils import vec3, float1, quaternion_from_euler


class SceneGeneratorSystem:
    @staticmethod
    def update(registry: Registry):
        for generator_entity, (generator_state, ) in registry.view(SceneGeneratorState):
            if not generator_state.should_generate:
                return

            generator_state.should_generate = False

            r_assets = registry.get_singleton(AssetsState)
            r_disposal = registry.get_singleton(Disposal)
            if not r_assets or not r_disposal:
                return

            _, (assets_state,) = r_assets
            _, (disposal,) = r_disposal

            children = registry.get_children(generator_entity)
            for child in children:
                should_dispose = True
                r_child = registry.get_components(child, EntityFlags)
                if r_child is not None:
                    (child_flags, ) = r_child
                    should_dispose = not child_flags.is_internal

                if should_dispose:
                    disposal.entities_to_dispose.add(child)

            mat_road = Material(
                albedo=vec3(0.1, 0.1, 0.1),
                roughness=float1(0.8),
                metallic=float1(0.0),
                reflectance=float1(0.1),
                ao=float1(0.1),
            )
            mat_sidewalk = Material(
                albedo=vec3(0.3, 0.3, 0.3),
                roughness=float1(0.7),
                metallic=float1(0.0),
                reflectance=float1(0.1),
                ao=float1(0.1),
            )
            mat_building = Material(
                albedo=vec3(0.5, 0.4, 0.3),
                roughness=float1(0.9),
                metallic=float1(0.0),
                reflectance=float1(0.1),
                ao=float1(0.1),
            )
            mat_vehicle = Material(
                albedo=vec3(0.8, 0.1, 0.1),
                roughness=float1(0.4),
                metallic=float1(0.5),
                reflectance=float1(0.5),
                ao=float1(0.1),
            )

            if generator_state.cube_mesh is None:
                cube_vertices, cube_indices = generate_cube(size=1.0)
                generator_state.cube_mesh = AssetSystem.create_immediate_mesh(assets_state, cube_vertices, cube_indices)

            if generator_state.plane_mesh is None:
                plane_vertices, plane_indices = generate_plane()
                generator_state.plane_mesh = AssetSystem.create_immediate_mesh(assets_state, plane_vertices, plane_indices)

            cube_mesh = generator_state.cube_mesh
            plane_mesh = generator_state.plane_mesh

            # == roads ==
            road = registry.create_entity()
            registry.add_components(
                road,
                EntityFlags(name="Road"),
                Transform(
                    position=vec3(0, 0, 0),
                    scale=vec3(generator_state.street_width, 1, generator_state.street_length)
                ),
                Visuals(plane_mesh, mat_road)
            )
            registry.set_parent(road, generator_entity)

            # == sidewalks ==
            sidewalk_height = 0.2
            sidewalk_width = generator_state.sidewalk_width + 20.0
            for side in [-1, 1]:
                sidewalk = registry.create_entity()
                x_pos = side * (generator_state.street_width / 2 + sidewalk_width / 2)
                registry.add_components(
                    sidewalk,
                    EntityFlags(
                        name=f"Sidewalk {'Left' if side < 0 else 'Right'}",
                        classification=EntityClassification.Environment
                    ),
                    Transform(
                        position=vec3(x_pos, sidewalk_height / 2, 0),
                        scale=vec3(sidewalk_width, sidewalk_height, generator_state.street_length)
                    ),
                    Visuals(cube_mesh, mat_sidewalk)
                )
                registry.set_parent(sidewalk, generator_entity)

            # == buildings ==
            for i in range(generator_state.building_count):
                side = random.choice([-1, 1])
                z_pos = random.uniform(-generator_state.street_length / 2, generator_state.street_length / 2)
                x_pos = side * (generator_state.street_width / 2 + generator_state.sidewalk_width + random.uniform(2, 10))

                width = random.uniform(generator_state.min_building_width, generator_state.max_building_width)
                depth = random.uniform(generator_state.min_building_width, generator_state.max_building_width)
                height = random.uniform(generator_state.min_building_height, generator_state.max_building_height)

                building = registry.create_entity()
                registry.add_components(
                    building,
                    EntityFlags(
                        name=f"Building {i}",
                        classification=EntityClassification.Building
                    ),
                    Transform(
                        position=vec3(x_pos, sidewalk_height + height / 2, z_pos),
                        scale=vec3(width, height, depth)
                    ),
                    Visuals(cube_mesh, mat_building)
                )
                registry.set_parent(building, generator_entity)

            # == vehicles ==
            for i in range(generator_state.vehicle_count):
                lane = random.choice([-1, 1])
                x_pos = lane * (generator_state.street_width / 4)
                z_pos = random.uniform(-generator_state.street_length / 2, generator_state.street_length / 2)

                speed = random.uniform(generator_state.min_vehicle_speed, generator_state.max_vehicle_speed)
                direction = vec3(0, 0, 1 if lane > 0 else -1)

                vehicle = registry.create_entity()
                registry.add_components(
                    vehicle,
                    EntityFlags(
                        name=f"Vehicle {i}",
                        classification=EntityClassification.Vehicle
                    ),
                    Transform(
                        position=vec3(x_pos, 0.5, z_pos),
                        scale=vec3(1.5, 1.0, 3.0),
                        rotation=quaternion_from_euler(vec3(0, 0 if lane > 0 else 180, 0))
                    ),
                    Visuals(cube_mesh, mat_vehicle),
                    Vehicle(speed=speed, direction=direction)
                )
                registry.set_parent(vehicle, generator_entity)
