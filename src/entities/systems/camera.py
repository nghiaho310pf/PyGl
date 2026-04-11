import math
import numpy as np
from imgui_bundle import imgui

from entities.components.camera import Camera
from entities.components.transform import Transform
from entities.components.camera_state import CameraState
from entities.registry import Registry
from math_utils import vec3
import math_utils


class CameraSystem:
    @staticmethod
    def update(registry: Registry, window_size: tuple[int, int], time: float, delta_time: float):
        r_camera_state = registry.get_singleton(CameraState)

        if r_camera_state is None:
            raise RuntimeError("CameraSystem is missing a CameraState singleton")
        camera_state_entity, (camera_state, ) = r_camera_state

        parent_entity = registry.get_parent(camera_state_entity)

        if parent_entity is None:
            r = registry.get_singleton(Transform, Camera)
            if r is None:
                return
            camera_entity, (camera_transform, camera) = r
            registry.set_parent(camera_state_entity, camera_entity)
        else:
            r = registry.get_components(parent_entity, Transform, Camera)
            if r is None:
                raise RuntimeError("CameraState singleton parented to an entity without (Transform, Camera)")
            (camera_transform, camera) = r

        io = imgui.get_io()

        if imgui.is_mouse_clicked(0) and not io.want_capture_mouse:
            camera_state.is_rotating = True
        if imgui.is_mouse_released(0):
            camera_state.is_rotating = False

        if imgui.is_mouse_clicked(2) and not io.want_capture_mouse:
            camera_state.is_panning = True
        if imgui.is_mouse_released(2):
            camera_state.is_panning = False

        if imgui.is_mouse_clicked(1) and not io.want_capture_mouse:
            camera_state.is_zooming = True
        if imgui.is_mouse_released(1):
            camera_state.is_zooming = False

        if camera_state.is_rotating:
            yaw_change = -io.mouse_delta.x * camera_state.rotation_speed
            pitch_change = io.mouse_delta.y * camera_state.rotation_speed

            new_pitch = camera_transform.rotation[0] - pitch_change
            new_yaw = camera_transform.rotation[1] - yaw_change
            new_pitch = max(-89.0, min(89.0, new_pitch))

            camera_transform.rotation = vec3(
                new_pitch, new_yaw, camera_transform.rotation[2])

        pitch_rad, yaw_rad, roll_rad = np.radians(camera_transform.rotation)

        front = np.array([
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad)
        ], dtype=np.float32)
        front /= np.linalg.norm(front)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(front, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, front) * math.cos(roll_rad) + \
            right * math.sin(roll_rad)
        up /= np.linalg.norm(up)

        if camera_state.is_panning:
            pan_amount = camera.focal_point_distance * camera_state.pan_speed
            pan_move = (right * -io.mouse_delta.x + up *
                        io.mouse_delta.y) * pan_amount
            camera_state.focal_point += pan_move

        if camera_state.is_zooming:
            zoom_factor = io.mouse_delta.y * camera_state.zoom_speed
            camera.focal_point_distance *= (1.0 + zoom_factor)

        if not io.want_capture_mouse and io.mouse_wheel != 0.0:
            zoom_factor = -io.mouse_wheel * camera_state.scroll_zoom_speed
            camera.focal_point_distance *= (1.0 + zoom_factor)

        camera.focal_point_distance = max(0.1, camera.focal_point_distance)

        is_input_active = (
            camera_state.is_rotating or
            camera_state.is_panning or
            camera_state.is_zooming or
            (not io.want_capture_mouse and io.mouse_wheel != 0.0)
        )

        if is_input_active:
            cam_pos = camera_state.focal_point - front * camera.focal_point_distance
            camera_transform.position = vec3(*cam_pos)
        else:
            camera_state.focal_point = np.array(camera_transform.position + front * camera.focal_point_distance, dtype=np.float32)

        window_width, window_height = window_size
        aspect_ratio = window_width / window_height

        camera_state.front = front
        camera_state.right = right
        camera_state.up = up
        target = camera_transform.position + front
        camera_state.view_matrix = math_utils.create_look_at(camera_transform.position, target, up)
        camera_state.projection_matrix = math_utils.create_perspective_projection(
            camera.fov, aspect_ratio, camera.near, camera.far
        )
        camera_state.view_projection_matrix = camera_state.view_matrix @ camera_state.projection_matrix
