import numpy as np
from imgui_bundle import imgui

from entities.components.camera import Camera
from entities.components.transform import Transform
from entities.components.camera_state import CameraState
from entities.components.ui.gizmo_state import GizmoState
from entities.registry import Registry
from math_utils import minimize_euler, vec3
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
            parent_entity, (camera_transform, camera) = r
            registry.set_parent(camera_state_entity, parent_entity)
        else:
            r = registry.get_components(parent_entity, Transform, Camera)
            if r is None:
                raise RuntimeError("CameraState singleton parented to an entity without (Transform, Camera)")
            (camera_transform, camera) = r

        if camera_state.last_camera_id != parent_entity:
            CameraSystem.sync_focal_point(camera_state, camera, camera_transform)
            camera_state.last_camera_id = parent_entity

        io = imgui.get_io()
        r_gizmo = registry.get_singleton(GizmoState)
        gizmo_dragging = r_gizmo[1][0].is_dragging if r_gizmo else False

        if gizmo_dragging:
            camera_state.is_rotating = False
            camera_state.is_panning = False
            camera_state.is_zooming = False

        if imgui.is_mouse_clicked(0) and not io.want_capture_mouse and not gizmo_dragging:
            camera_state.is_rotating = True
        if imgui.is_mouse_released(0):
            camera_state.is_rotating = False

        if imgui.is_mouse_clicked(2) and not io.want_capture_mouse and not gizmo_dragging:
            camera_state.is_panning = True
        if imgui.is_mouse_released(2):
            camera_state.is_panning = False

        if imgui.is_mouse_clicked(1) and not io.want_capture_mouse and not gizmo_dragging:
            camera_state.is_zooming = True
        if imgui.is_mouse_released(1):
            camera_state.is_zooming = False

        if camera_state.is_rotating:
            yaw_change = io.mouse_delta.x * camera_state.rotation_speed
            pitch_change = io.mouse_delta.y * camera_state.rotation_speed

            cam_rot_matrix = math_utils.create_transformation_matrix(
                vec3(0, 0, 0),
                camera_transform.local.rotation,
                vec3(1, 1, 1)
            )
            forward = -cam_rot_matrix[0:3, 2]
            world_up = vec3(0.0, 1.0, 0.0)

            if abs(forward[1]) > 0.9999:
                horizontal_right = cam_rot_matrix[0:3, 0]
            else:
                horizontal_right = math_utils.normalize(np.cross(forward, world_up))

            current_pitch = np.degrees(np.arcsin(np.clip(forward[1], -1.0, 1.0)))
            new_pitch = np.clip(current_pitch - pitch_change, -89.0, 89.0)
            actual_pitch_change = new_pitch - current_pitch

            q_yaw = math_utils.quaternion_from_axis_angle(world_up, -yaw_change)
            q_pitch = math_utils.quaternion_from_axis_angle(horizontal_right, actual_pitch_change)

            new_rot = math_utils.quaternion_mul(q_pitch, camera_transform.local.rotation)
            new_rot = math_utils.quaternion_mul(q_yaw, new_rot)

            camera_transform.local.rotation = math_utils.normalize(new_rot)  # type: ignore

        # recalculate since the code directly above us definitely changed the rotation
        cam_rot_matrix = math_utils.create_transformation_matrix(
            vec3(0, 0, 0),
            camera_transform.local.rotation,
            vec3(1, 1, 1)
        )

        right = cam_rot_matrix[0:3, 0]
        up = cam_rot_matrix[0:3, 1]
        backward = cam_rot_matrix[0:3, 2]
        front = -backward

        if camera_state.is_panning:
            pan_amount = camera.focal_point_distance * camera_state.pan_speed
            pan_move = (right * -io.mouse_delta.x + up * io.mouse_delta.y) * pan_amount
            camera_state.focal_point += np.astype(pan_move, np.float32)

        if camera_state.is_zooming:
            zoom_factor = io.mouse_delta.y * camera_state.zoom_speed
            camera.focal_point_distance *= (1.0 + zoom_factor)

        if not io.want_capture_mouse and io.mouse_wheel != 0.0:
            zoom_factor = -io.mouse_wheel * camera_state.scroll_zoom_speed
            camera.focal_point_distance *= (1.0 + zoom_factor)

        is_input_active = (
            camera_state.is_rotating or
            camera_state.is_panning or
            camera_state.is_zooming or
            (not io.want_capture_mouse and io.mouse_wheel != 0.0)
        )

        if is_input_active:
            camera.focal_point_distance = max(0.1, camera.focal_point_distance)
            camera_transform.local.position = camera_state.focal_point + (backward * camera.focal_point_distance)  # type: ignore
        else:
            CameraSystem.sync_focal_point(camera_state, camera, camera_transform)

        cam_world_matrix = math_utils.create_transformation_matrix(
            camera_transform.world.position,
            camera_transform.world.rotation,
            camera_transform.world.scale,
        )

        try:
            camera_state.view_matrix = np.linalg.inv(cam_world_matrix)
        except np.linalg.LinAlgError:
            return

        window_width, window_height = window_size
        aspect_ratio = window_width / window_height

        camera_state.camera_position = camera_transform.world.position
        camera_state.camera_near = camera.near
        camera_state.camera_far = camera.far
        camera_state.front = front
        camera_state.right = right
        camera_state.up = up
        camera_state.projection_matrix = math_utils.create_perspective_projection(
            camera.fov, aspect_ratio, camera.near, camera.far
        )
        camera_state.view_projection_matrix = camera_state.projection_matrix @ camera_state.view_matrix

    @staticmethod
    def sync_focal_point(state: CameraState, camera: Camera, transform: Transform):
        # sync based on world transform so focal point is in world space
        rot_matrix = math_utils.create_transformation_matrix(
            vec3(0, 0, 0), transform.world.rotation, vec3(1, 1, 1)
        )
        backward = math_utils.normalize(rot_matrix[0:3, 2])
        state.focal_point = transform.world.position - (backward * camera.focal_point_distance)  # type: ignore
