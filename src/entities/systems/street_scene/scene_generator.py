import random

from entities.registry import Registry
from entities.components.transform import Transform
from entities.components.visuals.visuals import Visuals
from entities.components.visuals.assets import AssetsState
from entities.components.visuals.material import Material
from entities.components.entity_flags import EntityFlags
from entities.components.disposal import Disposal
from entities.components.street_scene.scene_generator_state import SceneGeneratorState
from entities.components.street_scene.vehicle import Vehicle
from entities.components.street_scene.building import Building
from entities.components.street_scene.environment import Environment
from entities.systems.assets import AssetSystem
from meshes.volumes.cube import generate_cube
from meshes.surfaces.plane import generate_plane
from math_utils import vec3, float1, quaternion_from_euler, quaternion_identity


from entities.components.spawner_state import SpawnerState
from entities.systems.spawner import SpawnerSystem


class SceneGeneratorSystem:
    @staticmethod
    def update(registry: Registry):
        for generator_entity, (generator_state, ) in registry.view(SceneGeneratorState):
            if not generator_state.should_generate:
                return

            generator_state.should_generate = False

            r_assets = registry.get_singleton(AssetsState)
            r_disposal = registry.get_singleton(Disposal)
            r_spawner = registry.get_singleton(SpawnerState)
            if not r_assets or not r_disposal or not r_spawner:
                return

            _, (assets_state,) = r_assets
            _, (disposal,) = r_disposal
            _, (spawner_state,) = r_spawner

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

            if generator_state.cube_mesh is None:
                cube_vertices, cube_indices = generate_cube(size=1.0)
                generator_state.cube_mesh = AssetSystem.create_immediate_mesh(assets_state, cube_vertices, cube_indices)

            if generator_state.plane_mesh is None:
                plane_vertices, plane_indices = generate_plane()
                generator_state.plane_mesh = AssetSystem.create_immediate_mesh(assets_state, plane_vertices, plane_indices)

            if generator_state.car_model is None:
                generator_state.car_model = AssetSystem.request_model(assets_state, "assets/car/1399 Taxi.obj")
            if generator_state.bus_model is None:
                generator_state.bus_model = AssetSystem.request_model(assets_state, "assets/bus/bus_merged.obj")

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
                Visuals(plane_mesh, mat_road),
                Environment()
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
                        name=f"Sidewalk {'Left' if side < 0 else 'Right'}"
                    ),
                    Transform(
                        position=vec3(x_pos, sidewalk_height / 2, 0),
                        scale=vec3(sidewalk_width, sidewalk_height, generator_state.street_length)
                    ),
                    Visuals(cube_mesh, mat_sidewalk),
                    Environment()
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
                        name=f"Building {i}"
                    ),
                    Transform(
                        position=vec3(x_pos, sidewalk_height + height / 2, z_pos),
                        scale=vec3(width, height, depth)
                    ),
                    Visuals(cube_mesh, mat_building),
                    Building()
                )
                registry.set_parent(building, generator_entity)

            # == vehicles ==
            lanes_count = generator_state.lanes_per_direction
            lane_width = (generator_state.street_width / 2) / lanes_count

            for i in range(generator_state.vehicle_count):
                side = random.choice([-1, 1])
                lane_index = random.randint(0, lanes_count - 1)

                # center of the lane
                x_pos = side * (lane_index * lane_width + lane_width / 2.0)
                z_pos = random.uniform(-generator_state.street_length / 2, generator_state.street_length / 2)

                direction = vec3(0, 0, 1.0 if side > 0 else -1.0)
                lane_id = side * (lane_index + 1)

                # randomize vehicle type
                is_bus = random.random() < 0.2 # 20% chance of bus

                if is_bus:
                    model = generator_state.bus_model
                    model_scale = vec3(1.0, 1.0, 1.0)
                    model_offset = quaternion_identity()
                    speed = random.uniform(generator_state.min_vehicle_speed, (generator_state.min_vehicle_speed + generator_state.max_vehicle_speed) / 2)
                    max_accel = random.uniform(0.5, 1.5)
                    max_brake = random.uniform(2.0, 4.0)
                else:
                    model = generator_state.car_model
                    model_scale = vec3(0.04, 0.04, 0.04)
                    model_offset = quaternion_identity()
                    speed = random.uniform(generator_state.min_vehicle_speed, generator_state.max_vehicle_speed)
                    max_accel = random.uniform(1.0, 3.0)
                    max_brake = random.uniform(4.0, 8.0)

                # base directional rotation (facing forward or backward along Z)
                dir_rot = quaternion_from_euler(vec3(0, 0 if side > 0 else 180, 0))

                vehicle_entity = registry.create_entity()
                registry.add_components(
                    vehicle_entity,
                    EntityFlags(
                        name=f"{'Bus' if is_bus else 'Vehicle'} {i}"
                    ),
                    Transform(
                        position=vec3(x_pos, 0.0, z_pos),
                        scale=vec3(1, 1, 1),
                        rotation=dir_rot.copy()
                    ),
                    Vehicle(
                        target_speed=speed,
                        current_speed=speed,
                        direction=direction,
                        lane_id=lane_id,
                        target_lane_id=lane_id,
                        base_rotation=dir_rot,
                        max_acceleration=max_accel,
                        max_braking=max_brake
                    )
                )
                registry.set_parent(vehicle_entity, generator_entity)

                visual_root = registry.create_entity()
                registry.add_components(
                    visual_root,
                    EntityFlags(name="Visuals"),
                    Transform(
                        scale=model_scale,
                        rotation=model_offset
                    )
                )
                registry.set_parent(visual_root, vehicle_entity)

                if model:
                    SpawnerSystem.load_and_spawn_one(
                        spawner_state,
                        model,
                        parent_entity=visual_root
                    )
