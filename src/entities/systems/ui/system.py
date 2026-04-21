from imgui_bundle import imgui, icons_fontawesome_6

from entities.components.camera_state import CameraState
from entities.components.disposal import Disposal
from entities.components.ui.icon_render_state import IconRenderState
from entities.components.ui.gizmo_state import GizmoState, GizmoMode
from entities.components.render_state import RenderState
from entities.components.transform import Transform
from entities.components.ui.ui_state import UiState
from entities.components.visuals.visuals import Visuals
from entities.components.visuals.assets import AssetStatus, AssetsState
from entities.components.spawner_state import SpawnerState
from entities.registry import Registry

from entities.systems.ui.creation import draw_creation_section
from entities.systems.ui.entity_list import draw_entity_list_section
from entities.systems.ui.inspector import draw_inspector_section
from entities.systems.ui.graphics import draw_graphics_section


class UiSystem:
    @staticmethod
    def update(registry: Registry, time: float, delta_time: float):
        r_ui = registry.get_singleton(UiState)
        r_admin = registry.get_singleton(AssetsState, SpawnerState, RenderState, IconRenderState, Disposal)
        r_camera_state = registry.get_singleton(CameraState)
        r_gizmo = registry.get_singleton(GizmoState)
        if r_ui is None or r_admin is None or r_camera_state is None or r_gizmo is None:
            return
        ui_entity, (ui_state, ) = r_ui
        admin_entity, (assets_state, spawner_state, render_state, icon_render_state, disposal) = r_admin
        camera_state_entity, (camera_state, ) = r_camera_state
        gizmo_state_entity, (gizmo_state, ) = r_gizmo

        r_preview = registry.get_components(ui_state.preview_entity, Transform, Visuals)
        if r_preview is None:
            return
        (preview_transform, preview_visuals) = r_preview

        selected_entity = registry.get_parent(ui_state.selection_child_entity)

        viewport = imgui.get_main_viewport()

        window_pos = (viewport.work_pos.x + viewport.work_size.x, viewport.work_pos.y)
        imgui.set_next_window_pos(window_pos, imgui.Cond_.always, pivot=(1.0, 0.0))
        imgui.set_next_window_size((350, viewport.work_size.y), imgui.Cond_.first_use_ever)

        min_size = (150, viewport.work_size.y)
        max_size = (viewport.work_size.x * 0.8, viewport.work_size.y)
        imgui.set_next_window_size_constraints(min_size, max_size)
        main_expanded, main_opened = imgui.begin("Scene Panel")
        if not main_expanded:
            imgui.end()
            return

        if selected_entity is None:
            imgui.begin_disabled()
        if imgui.button("Deselect"):
            registry.set_parent(ui_state.selection_child_entity, None)
        if selected_entity is None:
            imgui.end_disabled()
        imgui.same_line()
        if imgui.radio_button("Translate", gizmo_state.mode == GizmoMode.Translate):
            gizmo_state.mode = GizmoMode.Translate
        imgui.same_line()
        if imgui.radio_button("Rotate", gizmo_state.mode == GizmoMode.Rotate):
            gizmo_state.mode = GizmoMode.Rotate
        imgui.separator()

        # warn when there's no camera
        if registry.get_parent(camera_state_entity) is None:
            imgui.text_colored((1.0, 0.8, 0.0, 1.0), f"{icons_fontawesome_6.ICON_FA_TRIANGLE_EXCLAMATION} No camera.")

        loading_model_count = sum(1 for m in assets_state.models.values() if m.status in (AssetStatus.Unloaded, AssetStatus.Loading))
        loading_mesh_count = sum(1 for m in assets_state.meshes.values() if m.status in (AssetStatus.Unloaded, AssetStatus.Loading))
        loading_texture_count = sum(1 for m in assets_state.textures.values() if m.status in (AssetStatus.Unloaded, AssetStatus.Loading))

        if loading_model_count > 0:
            imgui.text_colored((1.0, 0.8, 0.0, 1.0), f"{icons_fontawesome_6.ICON_FA_TRIANGLE_EXCLAMATION} {loading_model_count} model(s) are loading.")
        if loading_mesh_count > 0:
            imgui.text_colored((1.0, 0.8, 0.0, 1.0), f"{icons_fontawesome_6.ICON_FA_TRIANGLE_EXCLAMATION} {loading_mesh_count} meshes(s) are loading.")
        if loading_texture_count > 0:
            imgui.text_colored((1.0, 0.8, 0.0, 1.0), f"{icons_fontawesome_6.ICON_FA_TRIANGLE_EXCLAMATION} {loading_texture_count} textures(s) are loading.")

        draw_creation_section(
            registry, ui_state, assets_state, spawner_state,
            camera_state, preview_transform, preview_visuals
        )

        draw_entity_list_section(
            registry, ui_state, selected_entity
        )

        draw_inspector_section(
            registry, selected_entity, ui_state, assets_state,
            camera_state_entity, camera_state, disposal
        )

        draw_graphics_section(
            render_state, icon_render_state
        )

        imgui.end()
