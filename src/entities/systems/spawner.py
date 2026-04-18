from entities.registry import Registry
from entities.components.spawner_state import SpawnRequest, SpawnerState
from entities.components.visuals.assets import AssetsState, AssetStatus, ModelAsset
from entities.components.transform import Transform
from entities.components.entity_flags import EntityFlags
from entities.components.visuals.visuals import Visuals
from entities.components.visuals.material import Material
from entities.systems.assets import AssetSystem
import math_utils


class SpawnerSystem:
    @staticmethod
    def update(registry: Registry):
        r_spawner = registry.get_singleton(SpawnerState)
        if r_spawner is None: return
        spawner_entity, (spawner_state, ) = r_spawner

        r_assets = registry.get_singleton(AssetsState)
        if r_assets is None: return
        assets_entity, (assets_state, ) = r_assets

        instantiations = []
        for i in range(len(spawner_state.pending_spawns) - 1, -1, -1):
            request = spawner_state.pending_spawns[i]

            if request.model.status == AssetStatus.Ready:
                instantiations.append((request.model, request.root_transform))
                spawner_state.pending_spawns.pop(i)
            elif request.model.status == AssetStatus.Failed:
                spawner_state.pending_spawns.pop(i)

        for (model, root_transform) in instantiations:
            SpawnerSystem.instantiate_model(registry, assets_state, model, root_transform)

    @staticmethod
    def load_and_spawn_one(spawner_state: SpawnerState, assets_state: AssetsState, filepath: str, transform: Transform | None = None):
        if transform is None:
            transform = Transform()
        model_asset = AssetSystem.request_model(assets_state, filepath)
        spawner_state.pending_spawns.append(SpawnRequest(model_asset, transform))

    @staticmethod
    def instantiate_model(registry: Registry, assets_state: AssetsState, model: ModelAsset, root_transform: Transform) -> list[int]:
        if model.status != AssetStatus.Ready:
            print(f"Warning: Attempted to spawn model {model.filepath} before it was Ready.")
            return []

        spawned_entities = []

        for node in model.nodes:
            entity = registry.create_entity()

            world_pos = root_transform.position + node.local_position
            world_rot = math_utils.quaternion_mul(root_transform.rotation, node.local_rotation)
            world_scale = root_transform.scale * node.local_scale

            node_transform = Transform(position=world_pos, rotation=world_rot, scale=world_scale)

            mesh = AssetSystem.request_mesh(assets_state, node.mesh_id)
            mat = Material(
                albedo=node.material_template.albedo,
                roughness=node.material_template.roughness,
                metallic=node.material_template.metallic,
                reflectance=node.material_template.reflectance,
                translucency=node.material_template.translucency,
                ao=node.material_template.ao,
            )

            if node.material_template.albedo_map_id is not None:
                mat.albedo_map = AssetSystem.request_texture(assets_state, node.material_template.albedo_map_id)

            if node.material_template.normal_map_id is not None:
                mat.normal_map = AssetSystem.request_texture(assets_state, node.material_template.normal_map_id)

            if node.material_template.roughness_map_id is not None:
                mat.roughness_map = AssetSystem.request_texture(assets_state, node.material_template.roughness_map_id)

            if node.material_template.metallic_map_id is not None:
                mat.metallic_map = AssetSystem.request_texture(assets_state, node.material_template.metallic_map_id)

            registry.add_components(
                entity,
                EntityFlags(name=f"{model.filepath.split('/')[-1]}_{node.name}"),
                node_transform,
                Visuals(mesh=mesh, material=mat)
            )

            spawned_entities.append(entity)

        return spawned_entities
