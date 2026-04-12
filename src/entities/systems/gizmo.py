import math
import numpy as np
import numpy.typing as npt
from imgui_bundle import imgui

from entities.components.camera_state import CameraState
from entities.components.ui.ui_state import UiState
from entities.components.ui.gizmo_state import GizmoState, GizmoMode, GizmoAxis
from entities.components.transform import Transform
from entities.registry import Registry
from math_utils import (
    vec2, unpack_vec2, vec3, vec4, normalize,
    quaternion_from_axis_angle, quaternion_mul
)


class GizmoSystem:
    @staticmethod
    def update(registry: Registry, window_size: tuple[int, int]):
        r_admin = registry.get_singleton(GizmoState, UiState)
        if r_admin is None:
            return
        _, (gizmo_state, ui_state) = r_admin

        r_camera_state = registry.get_singleton(CameraState)
        if r_camera_state is None:
            return
        _, (camera_state, ) = r_camera_state

        selected_entity = registry.get_parent(ui_state.selection_child_entity)
        if selected_entity is None:
            gizmo_state.is_dragging = False
            gizmo_state.active_axis = GizmoAxis.NoAxis
            return

        r_transform = registry.get_components(selected_entity, Transform)
        if r_transform is None:
            return
        (transform, ) = r_transform

        io = imgui.get_io()
        draw_list = imgui.get_background_draw_list()

        world_space_origin = transform.position
        distance_to_camera = np.linalg.norm(world_space_origin - camera_state.camera_position)
        gizmo_too_close = distance_to_camera < camera_state.camera_near

        screen_space_origin = GizmoSystem._world_to_screen(world_space_origin, camera_state.view_projection_matrix, window_size)

        if screen_space_origin is None:
            return

        world_space_position_vec4 = vec4(world_space_origin[0], world_space_origin[1], world_space_origin[2], 1.0)
        clip_space_position = camera_state.view_projection_matrix @ world_space_position_vec4
        depth = clip_space_position[3]

        if depth <= 0:
            # object is behind camera
            return

        # scale the gizmo up so it maintains a constant size in 2D pixel space on the screen
        world_scale = depth * 0.002 * (gizmo_state.gizmo_size / 80.0)

        axes = {
            GizmoAxis.X: vec3(1, 0, 0),
            GizmoAxis.Y: vec3(0, 1, 0),
            GizmoAxis.Z: vec3(0, 0, 1)
        }

        colors = {
            GizmoAxis.X: (1.0, 0.2, 0.2, 1.0),
            GizmoAxis.Y: (0.2, 1.0, 0.2, 1.0),
            GizmoAxis.Z: (0.2, 0.2, 1.0, 1.0)
        }

        mouse_position = vec2(io.mouse_pos.x, io.mouse_pos.y)

        # == handle hovering & mouse-down ==
        if not gizmo_state.is_dragging and not gizmo_too_close:
            gizmo_state.active_axis = GizmoAxis.NoAxis
            if not io.want_capture_mouse:
                # let's detect which axis is being hovered
                gizmo_state.active_axis = GizmoAxis.NoAxis

                if gizmo_state.mode == GizmoMode.Translate:
                    best_dist = 10.0
                    for axis_type, axis_direction in axes.items():
                        world_space_arrow_end = world_space_origin + axis_direction * (world_scale * 50.0)
                        screen_space_arrow_end = GizmoSystem._world_to_screen(world_space_arrow_end, camera_state.view_projection_matrix, window_size)
                        if screen_space_arrow_end is None:
                            continue

                        distance = GizmoSystem._point_to_segment_distance(mouse_position, screen_space_origin, screen_space_arrow_end)
                        if distance < best_dist:
                            best_dist = distance
                            gizmo_state.active_axis = axis_type

                elif gizmo_state.mode == GizmoMode.Rotate:
                    best_dist = 6.0
                    for axis_type in axes:
                        points = GizmoSystem._get_rotation_circle_points(world_space_origin, axis_type, camera_state, window_size, world_scale * 50.0)
                        if len(points) < 1:
                            # none of the points were visible
                            continue

                        for i in range(len(points) - 1):
                            d = GizmoSystem._point_to_segment_distance(mouse_position, points[i], points[i+1])
                            if d < best_dist:
                                best_dist = d
                                gizmo_state.active_axis = axis_type

            if gizmo_state.active_axis != GizmoAxis.NoAxis and imgui.is_mouse_clicked(0):
                gizmo_state.is_dragging = True
                gizmo_state.initial_mouse_position = mouse_position

                if gizmo_state.mode == GizmoMode.Translate:
                    gizmo_state.initial_transform_value = transform.position.copy()

                    axis_direction = axes[gizmo_state.active_axis]
                    initial_offset = GizmoSystem._get_ray_axis_intersection(mouse_position, gizmo_state.initial_transform_value, axis_direction, camera_state, window_size)
                    gizmo_state.initial_translation_axis_offset = initial_offset if initial_offset is not None else 0.0

                elif gizmo_state.mode == GizmoMode.Rotate:
                    gizmo_state.initial_transform_value = transform.rotation.copy()
                    gizmo_state.initial_rotation_angle = GizmoSystem._calculate_rotation_angle(mouse_position, screen_space_origin)

        # == handle active dragging ==
        if gizmo_state.is_dragging:
            if imgui.is_mouse_released(0):
                gizmo_state.is_dragging = False
            else:
                if gizmo_state.mode == GizmoMode.Translate:
                    GizmoSystem._apply_translation(gizmo_state, transform, camera_state, window_size, mouse_position)
                elif gizmo_state.mode == GizmoMode.Rotate:
                    GizmoSystem._apply_rotation(gizmo_state, transform, camera_state, window_size, mouse_position)

        # == render the visual gizmo ==
        if not gizmo_too_close:
            for axis_type, axis_direction in axes.items():
                is_active = gizmo_state.active_axis == axis_type
                color = colors[axis_type]
                if is_active:
                    color = (1.0, 1.0, 0.0, 1.0)

                color_u32 = imgui.get_color_u32(color)
                thickness = 3.0 if is_active else 2.0

                if gizmo_state.mode == GizmoMode.Translate:
                    world_space_arrow_end = world_space_origin + axis_direction * (world_scale * 50.0)
                    screen_space_arrow_end = GizmoSystem._world_to_screen(world_space_arrow_end, camera_state.view_projection_matrix, window_size)
                    if screen_space_arrow_end is None: continue

                    draw_list.add_line(
                        unpack_vec2(screen_space_origin),
                        unpack_vec2(screen_space_arrow_end),
                        color_u32,
                        thickness
                    )

                    screen_space_arrow_direction = screen_space_arrow_end - screen_space_origin
                    screen_space_arrow_length = np.linalg.norm(screen_space_arrow_direction)
                    if screen_space_arrow_length > 0.001:
                        GizmoSystem._draw_arrow_head(draw_list, screen_space_arrow_end, screen_space_arrow_direction / screen_space_arrow_length, color_u32)

                elif gizmo_state.mode == GizmoMode.Rotate:
                    points = GizmoSystem._get_rotation_circle_points(world_space_origin, axis_type, camera_state, window_size, world_scale * 50.0)
                    if len(points) > 1:
                        for i in range(len(points) - 1):
                            draw_list.add_line(
                                unpack_vec2(points[i]),
                                unpack_vec2(points[i+1]),
                                color_u32,
                                thickness
                            )

    @staticmethod
    def _world_to_screen(world_pos, view_projection_matrix, window_size):
        world_position_vec4 = vec4(world_pos[0], world_pos[1], world_pos[2], 1.0)
        clip_pos = view_projection_matrix @ world_position_vec4

        if clip_pos[3] <= 0:
            # point is behind camera
            return None

        ndc = clip_pos[:3] / clip_pos[3]

        window_width, window_height = window_size
        screen_x = (ndc[0] + 1.0) * 0.5 * window_width
        screen_y = (1.0 - ndc[1]) * 0.5 * window_height
        return vec2(screen_x, screen_y)

    @staticmethod
    def _get_ray_axis_intersection(mouse_position, origin, axis_direction, camera_state: CameraState, window_size):
        window_width, window_height = window_size

        # == compute the world-space ray that the mouse cursor casts from the camera ==
        ndc_mouse_x = (mouse_position[0] / window_width) * 2.0 - 1.0
        ndc_mouse_y = 1.0 - (mouse_position[1] / window_height) * 2.0

        ndc_space_ray_near = vec4(ndc_mouse_x, ndc_mouse_y, -1.0, 1.0)
        ndc_space_ray_far = vec4(ndc_mouse_x, ndc_mouse_y, 1.0, 1.0)

        inverse_view_projection_matrix = np.linalg.inv(camera_state.view_projection_matrix)

        world_space_ray_near = inverse_view_projection_matrix @ ndc_space_ray_near
        world_space_ray_far = inverse_view_projection_matrix @ ndc_space_ray_far
        # undo the perspective divide, trim to vec3
        world_space_ray_near = world_space_ray_near[:3] / world_space_ray_near[3]
        world_space_ray_far = world_space_ray_far[:3] / world_space_ray_far[3]

        ray_direction = normalize(world_space_ray_far - world_space_ray_near)

        # == compute a plane containing the axis and facing the camera ==
        axis_cross_front = np.cross(axis_direction, camera_state.front)
        axis_cross_front_length = np.linalg.norm(axis_cross_front)

        if axis_cross_front_length < 0.001:
            # fallback for when camera is looking down the axis
            optimal_plane_normal = camera_state.up
        else:
            # the ideal plane normal is perpendicular to both the axis and `axis_cross_front`,
            # i.e. the plane contains both the axis and `axis_cross_front`
            optimal_plane_normal = normalize(np.cross(axis_cross_front, axis_direction))

        # soft-clamp the movement as camera comes closer to looking down the axis
        soft_clamp_threshold = 0.8
        if axis_cross_front_length < soft_clamp_threshold:
            weight = axis_cross_front_length / soft_clamp_threshold
            plane_normal = normalize(optimal_plane_normal * weight + camera_state.front * (1.0 - weight))
        else:
            plane_normal = optimal_plane_normal

        # == intersect the plane and the ray ==
        intersection_denominator = np.dot(ray_direction, plane_normal)
        if abs(intersection_denominator) < 0.0001:
            # ray is parallel to the plane
            return None

        distance_along_ray = np.dot(origin - world_space_ray_near, plane_normal) / intersection_denominator
        intersection_point = world_space_ray_near + distance_along_ray * ray_direction

        # == project the hit point onto the axis ==
        distance_along_axis = np.dot(intersection_point - origin, axis_direction)
        return distance_along_axis

    @staticmethod
    def _point_to_segment_distance(p, a, b):
        ab = b - a
        ap = p - a
        len_sq = np.dot(ab, ab)

        if len_sq < 0.001:
            return np.linalg.norm(p - a)

        # project p onto ab
        t = np.dot(ap, ab) / len_sq
        t = np.clip(t, 0.0, 1.0)
        projected = a + t * ab

        return np.linalg.norm(p - projected)

    @staticmethod
    def _draw_arrow_head(draw_list, end, direction, color_u32):
        sideways = vec2(-direction[1], direction[0])
        head_length = 10.0
        head_width = 6.0

        arrow_tip = end + direction * (head_length * 0.5)
        arrow_left = end - direction * (head_length * 0.5) + sideways * head_width
        arrow_right = end - direction * (head_length * 0.5) - sideways * head_width

        draw_list.add_triangle_filled(
            unpack_vec2(arrow_tip),
            unpack_vec2(arrow_left),
            unpack_vec2(arrow_right),
            color_u32
        )

    @staticmethod
    def _get_rotation_circle_points(world_pos, axis_type, camera_state, window_size, world_radius):
        u, v = {
            GizmoAxis.X: (vec3(0, 1, 0), vec3(0, 0, 1)),
            GizmoAxis.Y: (vec3(1, 0, 0), vec3(0, 0, 1)),
            GizmoAxis.Z: (vec3(1, 0, 0), vec3(0, 1, 0))
        }[axis_type]

        segments = 64
        points: list[npt.NDArray] = []
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            p_world = world_pos + (u * math.cos(angle) + v * math.sin(angle)) * world_radius
            p_screen = GizmoSystem._world_to_screen(p_world, camera_state.view_projection_matrix, window_size)
            if p_screen is not None:
                points.append(p_screen)
        return points

    @staticmethod
    def _apply_translation(gizmo_state: GizmoState, transform: Transform, camera_state: CameraState, window_size: tuple[int, int], mouse_pos: npt.NDArray):
        axis_dir = {
            GizmoAxis.X: vec3(1, 0, 0),
            GizmoAxis.Y: vec3(0, 1, 0),
            GizmoAxis.Z: vec3(0, 0, 1)
        }[gizmo_state.active_axis]

        current_offset = GizmoSystem._get_ray_axis_intersection(mouse_pos, gizmo_state.initial_transform_value, axis_dir, camera_state, window_size)
        if current_offset is None:
            return

        world_delta = np.clip(current_offset - gizmo_state.initial_translation_axis_offset, -1500.0, 1500.0)
        transform.position = gizmo_state.initial_transform_value + axis_dir * world_delta

    @staticmethod
    def _calculate_rotation_angle(mouse_pos, center_screen):
        rel = mouse_pos - center_screen
        return math.atan2(rel[1], rel[0])

    @staticmethod
    def _apply_rotation(gizmo_state: GizmoState, transform: Transform, camera_state: CameraState, window_size: tuple[int, int], mouse_pos: npt.NDArray):
        center_screen = GizmoSystem._world_to_screen(transform.position, camera_state.view_projection_matrix, window_size)
        if center_screen is None:
            return

        current_angle = GizmoSystem._calculate_rotation_angle(mouse_pos, center_screen)
        delta_angle_rad = current_angle - gizmo_state.initial_rotation_angle
        delta_angle_deg = math.degrees(delta_angle_rad)

        world_axis = {
            GizmoAxis.X: vec3(1, 0, 0),
            GizmoAxis.Y: vec3(0, 1, 0),
            GizmoAxis.Z: vec3(0, 0, 1)
        }[gizmo_state.active_axis]

        if np.dot(world_axis, camera_state.front) < 0:
            # we're viewing from behind the spin. invert the angle to correctly follow the mouse
            delta_angle_deg = -delta_angle_deg

        delta_q = quaternion_from_axis_angle(world_axis, delta_angle_deg)
        transform.rotation = quaternion_mul(delta_q, gizmo_state.initial_transform_value)
