import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6

from entities.components.camera_state import CameraState
from entities.components.ui.icon_render_state import IconRenderState
from entities.components.point_light import PointLight
from entities.components.ui.ui_state import UiState
from entities.components.camera import Camera
from entities.components.transform import Transform
from entities.components.render_state import RenderState
from entities.registry import Registry


class IconRenderSystem:
    @staticmethod
    def update(registry: Registry, window_size: tuple[int, int]):
        r_admin = registry.get_singleton(IconRenderState, RenderState, UiState)
        if r_admin is None:
            raise RuntimeError("IconRenderSystem is missing a (IconRenderState, RenderState, UiState) singleton")
        _, (icon_render_state, render_state, ui_state) = r_admin

        r_camera_state = registry.get_singleton(CameraState)
        if r_camera_state is None:
            raise RuntimeError("IconRenderSystem is missing a CameraState singleton")
        camera_state_entity, (camera_state, ) = r_camera_state

        camera_parent = registry.get_parent(camera_state_entity)
        selected_entity = registry.get_parent(ui_state.selection_child_entity)

        draw_list = imgui.get_background_draw_list()

        if icon_render_state.draw_camera_icons:
            for entity, (transform, camera) in registry.view(Transform, Camera):
                if entity == camera_parent:
                    continue
                IconRenderSystem._draw_entity_icon(
                    draw_list, transform, camera_state.view_projection_matrix, window_size,
                    entity == selected_entity, icons_fontawesome_6.ICON_FA_CAMERA,
                    np.array([1.0, 1.0, 1.0])
                )

        if icon_render_state.draw_light_icons:
            for entity, (transform, light) in registry.view(Transform, PointLight):
                if entity == camera_parent:
                    continue
                IconRenderSystem._draw_entity_icon(
                    draw_list, transform, camera_state.view_projection_matrix, window_size,
                    entity == selected_entity, icons_fontawesome_6.ICON_FA_LIGHTBULB,
                    light.color * 0.6 + np.array([1.0, 1.0, 1.0]) * 0.4
                )

    @staticmethod
    def _draw_entity_icon(
        draw_list: imgui.ImDrawList,
        transform: Transform, view_projection_matrix: np.ndarray,
        window_size: tuple[int, int],
        selected: bool, icon: str,
        color: np.ndarray
    ):
        world_pos = np.array(
            [transform.position[0], transform.position[1], transform.position[2], 1.0])
        clip_pos = world_pos @ view_projection_matrix

        if clip_pos[3] <= 0:
            return

        ndc = clip_pos[:3] / clip_pos[3]

        if abs(ndc[0]) > 1.0 or abs(ndc[1]) > 1.0:
            return

        window_width, window_height = window_size
        screen_x = (ndc[0] + 1.0) * 0.5 * window_width
        screen_y = (1.0 - ndc[1]) * 0.5 * window_height

        scale = 1.0 if selected else 0.8
        base_size = imgui.calc_text_size(icon)

        alpha = 1.0 if selected else 0.7
        icon_color_u32 = imgui.get_color_u32(
            (float(color[0]), float(color[1]), float(color[2]), alpha)
        )

        draw_list.add_text(
            imgui.get_font(),
            imgui.get_font_size() * scale,
            (
                screen_x - (base_size.x * scale) * 0.5,
                screen_y - (base_size.y * scale) * 0.5
            ),
            icon_color_u32,
            icon
        )
