import math
import numpy as np
from imgui_bundle import imgui

from entities.components.camera import Camera
from entities.components.transform import Transform
from entities.components.render_state import RenderState
from entities.components.camera_control_state import CameraControlState
from entities.registry import Registry
from math_utils import vec3


class CameraControlSystem:
    @staticmethod
    def update(registry: Registry, time: float, delta_time: float):
        r_ctrl = registry.get_singleton(CameraControlState)
        r_render = registry.get_singleton(RenderState)

        if r_ctrl is None or r_render is None:
            return

        _, (ctrl_state,) = r_ctrl
        _, (render_state,) = r_render

        camera_entity = render_state.target_camera
        if camera_entity is None:
            return

        r_cam = registry.get_components(camera_entity, Transform)
        if r_cam is None:
            return

        (transform, ) = r_cam
        io = imgui.get_io()

        if imgui.is_mouse_clicked(0) and not io.want_capture_mouse:
            ctrl_state.is_rotating = True
        if imgui.is_mouse_released(0):
            ctrl_state.is_rotating = False

        if imgui.is_mouse_clicked(2) and not io.want_capture_mouse:
            ctrl_state.is_panning = True
        if imgui.is_mouse_released(2):
            ctrl_state.is_panning = False

        if imgui.is_mouse_clicked(1) and not io.want_capture_mouse:
            ctrl_state.is_zooming = True
        if imgui.is_mouse_released(1):
            ctrl_state.is_zooming = False

        if ctrl_state.is_rotating:
            yaw_change = -io.mouse_delta.x * ctrl_state.rotation_speed
            pitch_change = io.mouse_delta.y * ctrl_state.rotation_speed

            new_pitch = transform.rotation[0] - pitch_change
            new_yaw = transform.rotation[1] - yaw_change
            new_pitch = max(-89.0, min(89.0, new_pitch))

            transform.rotation = vec3(
                new_pitch, new_yaw, transform.rotation[2])

        pitch_rad, yaw_rad, roll_rad = np.radians(transform.rotation)

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

        if ctrl_state.is_panning:
            pan_amount = ctrl_state.focal_point_distance * ctrl_state.pan_speed
            pan_move = (right * -io.mouse_delta.x + up *
                        io.mouse_delta.y) * pan_amount
            ctrl_state.focal_point += pan_move

        if ctrl_state.is_zooming:
            zoom_factor = io.mouse_delta.y * ctrl_state.zoom_speed
            ctrl_state.focal_point_distance *= (1.0 + zoom_factor)

        if not io.want_capture_mouse and io.mouse_wheel != 0.0:
            zoom_factor = -io.mouse_wheel * ctrl_state.scroll_zoom_speed
            ctrl_state.focal_point_distance *= (1.0 + zoom_factor)

        ctrl_state.focal_point_distance = max(0.1, ctrl_state.focal_point_distance)

        is_input_active = (
            ctrl_state.is_rotating or 
            ctrl_state.is_panning or 
            ctrl_state.is_zooming or 
            (not io.want_capture_mouse and io.mouse_wheel != 0.0)
        )

        if is_input_active:
            cam_pos = ctrl_state.focal_point - front * ctrl_state.focal_point_distance
            transform.position = vec3(*cam_pos)
        else:
            ctrl_state.focal_point = np.array(transform.position + front * ctrl_state.focal_point_distance, dtype=np.float32)
