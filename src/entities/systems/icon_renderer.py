import math
import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6

from entities.components.point_light import PointLight
import math_utils
from entities.components.camera import Camera
from entities.components.transform import Transform
from entities.components.render_state import RenderState
from entities.registry import Registry


class IconRendererSystem:
    @staticmethod
    def update(registry: Registry, window_size: tuple[int, int]):
        width, height = window_size
        if width <= 0 or height <= 0:
            return

        aspect_ratio = width / height

        r = registry.get_singleton(RenderState)
        if r is None:
            return
        _, (render_state, ) = r

        target_camera_entity = render_state.target_camera
        if target_camera_entity is None:
            r_cam = registry.get_singleton(Transform, Camera)
            if r_cam is None:
                return
            _, r_cam = r_cam
        else:
            r_cam = registry.get_components(target_camera_entity, Transform, Camera)
            if r_cam is None:
                return
        camera_transform, camera = r_cam

        pitch_rad, yaw_rad, roll_rad = np.radians(camera_transform.rotation)
        front = math_utils.normalize(np.array([
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad)
        ]))
        right = math_utils.normalize(
            np.cross(front, np.array([0.0, 1.0, 0.0])))
        up = math_utils.normalize(
            np.cross(right, front)) * math.cos(roll_rad) + right * math.sin(roll_rad)

        target = camera_transform.position + front
        view_matrix = math_utils.create_look_at(
            camera_transform.position, target, up)
        proj_matrix = math_utils.create_perspective_projection(
            camera.fov, aspect_ratio, camera.near, camera.far
        )

        view_proj = view_matrix @ proj_matrix

        draw_list = imgui.get_background_draw_list()

        for entity, (transform, camera) in registry.view(Transform, Camera):
            if entity == target_camera_entity:
                continue

            world_pos = np.array(
                [transform.position[0], transform.position[1], transform.position[2], 1.0])
            clip_pos = world_pos @ view_proj

            if clip_pos[3] <= 0:
                continue

            ndc = clip_pos[:3] / clip_pos[3]

            if abs(ndc[0]) > 1.0 or abs(ndc[1]) > 1.0:
                continue

            screen_x = (ndc[0] + 1.0) * 0.5 * width
            screen_y = (1.0 - ndc[1]) * 0.5 * height

            icon = icons_fontawesome_6.ICON_FA_CAMERA
            text_size = imgui.calc_text_size(icon)

            pos = (screen_x - text_size.x * 0.5, screen_y - text_size.y * 0.5)
            draw_list.add_text(pos, imgui.get_color_u32(imgui.Col_.text), icon)

        for entity, (transform, light) in registry.view(Transform, PointLight):
            if entity == target_camera_entity:
                continue

            world_pos = np.array(
                [transform.position[0], transform.position[1], transform.position[2], 1.0])
            clip_pos = world_pos @ view_proj

            if clip_pos[3] <= 0:
                continue

            ndc = clip_pos[:3] / clip_pos[3]

            if abs(ndc[0]) > 1.0 or abs(ndc[1]) > 1.0:
                continue

            screen_x = (ndc[0] + 1.0) * 0.5 * width
            screen_y = (1.0 - ndc[1]) * 0.5 * height

            icon = icons_fontawesome_6.ICON_FA_LIGHTBULB
            text_size = imgui.calc_text_size(icon)

            pos = (screen_x - text_size.x * 0.5, screen_y - text_size.y * 0.5)
            draw_list.add_text(pos, imgui.get_color_u32(imgui.Col_.text), icon)
