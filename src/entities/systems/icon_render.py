import math
import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6

from entities.components.camera_state import CameraState
from entities.components.icon_render_state import IconRenderState
from entities.components.point_light import PointLight
from entities.components.ui_state import UiState
import math_utils
from entities.components.camera import Camera
from entities.components.transform import Transform
from entities.components.render_state import RenderState
from entities.registry import Registry


class IconRenderSystem:
    @staticmethod
    def update(registry: Registry, window_size: tuple[int, int]):
        window_width, window_height = window_size

        r_admin = registry.get_singleton(IconRenderState, RenderState, CameraState, UiState)
        if r_admin is None:
            return
        _, (icon_render_state, render_state, camera_state, ui_state) = r_admin

        draw_list = imgui.get_background_draw_list()

        if icon_render_state.draw_camera_icons:
            for entity, (transform, camera) in registry.view(Transform, Camera):
                if entity == camera_state.target_camera:
                    continue

                world_pos = np.array(
                    [transform.position[0], transform.position[1], transform.position[2], 1.0])
                clip_pos = world_pos @ camera_state.view_projection_matrix

                if clip_pos[3] <= 0:
                    continue

                ndc = clip_pos[:3] / clip_pos[3]

                if abs(ndc[0]) > 1.0 or abs(ndc[1]) > 1.0:
                    continue

                screen_x = (ndc[0] + 1.0) * 0.5 * window_width
                screen_y = (1.0 - ndc[1]) * 0.5 * window_height

                icon = icons_fontawesome_6.ICON_FA_CAMERA
                text_size = imgui.calc_text_size(icon)

                pos = (screen_x - text_size.x * 0.5, screen_y - text_size.y * 0.5)
                selected = entity == ui_state.selected_entity
                color = imgui.Col_.text if selected else imgui.Col_.text_disabled
                draw_list.add_text(pos, imgui.get_color_u32(color), icon)

        if icon_render_state.draw_light_icons:
            for entity, (transform, light) in registry.view(Transform, PointLight):
                if entity == camera_state.target_camera:
                    continue

                world_pos = np.array(
                    [transform.position[0], transform.position[1], transform.position[2], 1.0])
                clip_pos = world_pos @ camera_state.view_projection_matrix

                if clip_pos[3] <= 0:
                    continue

                ndc = clip_pos[:3] / clip_pos[3]

                if abs(ndc[0]) > 1.0 or abs(ndc[1]) > 1.0:
                    continue

                screen_x = (ndc[0] + 1.0) * 0.5 * window_width
                screen_y = (1.0 - ndc[1]) * 0.5 * window_height

                icon = icons_fontawesome_6.ICON_FA_LIGHTBULB
                text_size = imgui.calc_text_size(icon)

                pos = (screen_x - text_size.x * 0.5, screen_y - text_size.y * 0.5)
                selected = entity == ui_state.selected_entity
                color = imgui.Col_.text if selected else imgui.Col_.text_disabled
                draw_list.add_text(pos, imgui.get_color_u32(color), icon)
