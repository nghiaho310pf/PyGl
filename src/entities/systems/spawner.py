from pathlib import Path

from entities.registry import Registry
from entities.components.spawner_state import SpawnRequest, SpawnerState
from entities.components.visuals.assets import AssetsState, AssetStatus, ModelAsset
from entities.components.transform import Transform, TransformData
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
                instantiations.append((request.model, request.root_transform, request.parent_entity))
                spawner_state.pending_spawns.pop(i)
            elif request.model.status == AssetStatus.Failed:
                spawner_state.pending_spawns.pop(i)

        for (model, root_transform, parent_entity) in instantiations:
            SpawnerSystem.instantiate_model(registry, assets_state, model, root_transform, parent_entity)

    @staticmethod
    def load_and_spawn_one(spawner_state: SpawnerState, model_asset: ModelAsset, transform: Transform | None = None, parent_entity: int | None = None):
        if transform is None:
            transform = Transform()
        spawner_state.pending_spawns.append(SpawnRequest(model_asset, transform, parent_entity))

    @staticmethod
    def instantiate_model(registry: Registry, assets_state: AssetsState, model: ModelAsset, root_transform: Transform, parent_entity: int | None = None) -> list[int]:
        if model.status != AssetStatus.Ready:
            print(f"Warning: Attempted to spawn model {model.filepath} before it was Ready.")
            return []

        spawned_entities = []

        for node in model.nodes:
            entity = registry.create_entity()

            # apply the root transform for loose instantiations
            world_pos = root_transform.local.position + node.local_position
            world_rot = math_utils.quaternion_mul(root_transform.local.rotation, node.local_rotation)
            world_scale = root_transform.local.scale * node.local_scale
            node_transform = Transform(local=TransformData(position=world_pos, rotation=world_rot, scale=world_scale))

            if parent_entity is not None:
                registry.set_parent(entity, parent_entity)

            mesh = AssetSystem.request_mesh(assets_state, node.mesh_id)
            mat = Material(
                albedo=node.material_template.albedo,
                roughness=node.material_template.roughness,
                metallic=node.material_template.metallic,
                reflectance=node.material_template.reflectance,
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
                EntityFlags(name=f"{Path(model.filepath).name}"),
                node_transform,
                Visuals(mesh=mesh, material=mat)
            )

            spawned_entities.append(entity)

        return spawned_entities
