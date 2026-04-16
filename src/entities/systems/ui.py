import copy
import dataclasses
from typing import Any, Type

import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6, portable_file_dialogs as pfd

from entities.components.camera import Camera
from entities.components.camera_state import CameraState
from entities.components.directional_light import DirectionalLight
from entities.components.disposal import Disposal
from entities.components.gd.optimizer_state import OptimizerAlgorithm, OptimizerState
from entities.components.gd.surface import GradientDescentSurface, LossFunctionType
from entities.components.spawner_state import SpawnerState
from entities.components.surface_function import CompilationStatus, SurfaceFunction
from entities.components.ui.icon_render_state import IconRenderState
from entities.components.ui.gizmo_state import GizmoState, GizmoMode
from entities.components.point_light import PointLight
from entities.components.render_state import GlobalDrawMode, RenderState
from entities.components.transform import Transform
from entities.components.ui.ui_state import AddType, UiState
from entities.components.visuals.visuals import DrawMode, Visuals
from entities.components.entity_flags import EntityFlags
from entities.components.visuals.assets import AssetStatus, AssetsState
from entities.systems.assets import AssetSystem
from entities.registry import Registry
from entities.systems.spawner import SpawnerSystem
from meshes.surfaces.arrow import generate_arrow
from meshes.surfaces.circle import generate_circle
from meshes.surfaces.ellipse import generate_ellipse
from meshes.surfaces.hexagon import generate_hexagon
from meshes.surfaces.pentagon import generate_pentagon
from meshes.surfaces.plane import generate_plane
from meshes.surfaces.star import generate_star
from meshes.surfaces.trapezoid import generate_trapezoid
from meshes.surfaces.triangle import generate_triangle
from meshes.volumes.cone import generate_cone
from meshes.volumes.cube import generate_cube
from meshes.volumes.cylinder import generate_cylinder
from meshes.volumes.prism import generate_prism
from meshes.volumes.subdivided_spheres import generate_icosphere, generate_tetrasphere
from meshes.volumes.tetrahedron import generate_tetrahedron
from meshes.volumes.torus import generate_torus
from meshes.volumes.uv_sphere import generate_uv_sphere
from math_utils import float1, minimize_euler, quaternions_from_euler, vec3, quaternion_identity, quaternion_to_euler


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

        # == creation section ==
        if ui_state.should_close_add_menu:
            imgui.set_next_item_open(False)
            ui_state.should_close_add_menu = False
        add_menu_expanded = imgui.collapsing_header("+ Add")

        if not add_menu_expanded:
            preview_visuals.enabled = False
        else:
            if imgui.button("Load model..."):
                dialog = pfd.open_file(
                    title=f"Select model file",
                    default_path="",
                    filters=["3D Model Files", "*.glb *.gltf *.obj *.ply", "All Files", "*"]
                )
                result = dialog.result()

                if result and len(result) > 0:
                    filepath = result[0]
                    SpawnerSystem.load_and_spawn_one(
                        spawner_state, assets_state, filepath,
                        Transform(scale=vec3(0.04, 0.04, 0.04)),
                    )
                    ui_state.should_close_add_menu = True

            preview_transform.position = vec3(*camera_state.focal_point)
            preview_transform.rotation = quaternion_identity()
            preview_transform.scale = vec3(1.0, 1.0, 1.0)

            changed_type = False

            if imgui.radio_button("Triangle", ui_state.add_mesh_type == AddType.Triangle):
                ui_state.add_mesh_type = AddType.Triangle
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Plane", ui_state.add_mesh_type == AddType.Plane):
                ui_state.add_mesh_type = AddType.Plane
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Pentagon", ui_state.add_mesh_type == AddType.Pentagon):
                ui_state.add_mesh_type = AddType.Pentagon
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Hexagon", ui_state.add_mesh_type == AddType.Hexagon):
                ui_state.add_mesh_type = AddType.Hexagon
                changed_type = True
            if imgui.radio_button("Circle", ui_state.add_mesh_type == AddType.Circle):
                ui_state.add_mesh_type = AddType.Circle
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Ellipse", ui_state.add_mesh_type == AddType.Ellipse):
                ui_state.add_mesh_type = AddType.Ellipse
                changed_type = True
            if imgui.radio_button("Trapezoid", ui_state.add_mesh_type == AddType.Trapezoid):
                ui_state.add_mesh_type = AddType.Trapezoid
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Star", ui_state.add_mesh_type == AddType.Star):
                ui_state.add_mesh_type = AddType.Star
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Arrow", ui_state.add_mesh_type == AddType.Arrow):
                ui_state.add_mesh_type = AddType.Arrow
                changed_type = True

            if imgui.radio_button("Cube", ui_state.add_mesh_type == AddType.Cube):
                ui_state.add_mesh_type = AddType.Cube
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Tetrahedron", ui_state.add_mesh_type == AddType.Tetrahedron):
                ui_state.add_mesh_type = AddType.Tetrahedron
                changed_type = True

            if imgui.radio_button("Function surface", ui_state.add_mesh_type == AddType.FunctionSurface):
                ui_state.add_mesh_type = AddType.FunctionSurface
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Gradient descent surface", ui_state.add_mesh_type == AddType.GradientDescentSurface):
                ui_state.add_mesh_type = AddType.GradientDescentSurface
                changed_type = True

            if imgui.radio_button("Prism", ui_state.add_mesh_type == AddType.Prism):
                ui_state.add_mesh_type = AddType.Prism
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Cone", ui_state.add_mesh_type == AddType.Cone):
                ui_state.add_mesh_type = AddType.Cone
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Cylinder", ui_state.add_mesh_type == AddType.Cylinder):
                ui_state.add_mesh_type = AddType.Cylinder
                changed_type = True

            if imgui.radio_button("UV sphere", ui_state.add_mesh_type == AddType.UVSphere):
                ui_state.add_mesh_type = AddType.UVSphere
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Tetrasphere", ui_state.add_mesh_type == AddType.Tetrasphere):
                ui_state.add_mesh_type = AddType.Tetrasphere
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Icosphere", ui_state.add_mesh_type == AddType.Icosphere):
                ui_state.add_mesh_type = AddType.Icosphere
                changed_type = True

            if imgui.radio_button("Torus", ui_state.add_mesh_type == AddType.Torus):
                ui_state.add_mesh_type = AddType.Torus
                changed_type = True

            if imgui.radio_button("Directional light", ui_state.add_mesh_type == AddType.DirectionalLight):
                ui_state.add_mesh_type = AddType.DirectionalLight
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Point light", ui_state.add_mesh_type == AddType.PointLight):
                ui_state.add_mesh_type = AddType.PointLight
                changed_type = True
            imgui.same_line()
            if imgui.radio_button("Camera", ui_state.add_mesh_type == AddType.Camera):
                ui_state.add_mesh_type = AddType.Camera
                changed_type = True

            mesh_changed = not ui_state.preview_visual_initialized or changed_type

            if ui_state.add_mesh_type == AddType.Triangle:
                changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
                mesh_changed = mesh_changed or changed_s
            elif ui_state.add_mesh_type == AddType.Plane:
                changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
                mesh_changed = mesh_changed or changed_s
            elif ui_state.add_mesh_type == AddType.Pentagon:
                changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
                mesh_changed = mesh_changed or changed_s
            elif ui_state.add_mesh_type == AddType.Hexagon:
                changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
                mesh_changed = mesh_changed or changed_s
            elif ui_state.add_mesh_type == AddType.Circle:
                changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
                changed_se, ui_state.round_surface_sides = imgui.slider_int("Sides", ui_state.round_surface_sides, 3, 50)
                mesh_changed = mesh_changed or changed_s or changed_se
            elif ui_state.add_mesh_type == AddType.Ellipse:
                changed_rx, ui_state.ellipse_radius_x = imgui.drag_float("X radius", ui_state.ellipse_radius_x, 0.01, 0.01, 10.0)
                changed_rz, ui_state.ellipse_radius_z = imgui.drag_float("Z radius", ui_state.ellipse_radius_z, 0.01, 0.01, 10.0)
                changed_se, ui_state.round_surface_sides = imgui.slider_int("Sides", ui_state.round_surface_sides, 3, 50)
                mesh_changed = mesh_changed or changed_rx or changed_rz or changed_se
            elif ui_state.add_mesh_type == AddType.Ellipse:
                changed_rx, ui_state.ellipse_radius_x = imgui.drag_float("X radius", ui_state.ellipse_radius_x, 0.01, 0.01, 10.0)
                changed_rz, ui_state.ellipse_radius_z = imgui.drag_float("Z radius", ui_state.ellipse_radius_z, 0.01, 0.01, 10.0)
                changed_se, ui_state.round_surface_sides = imgui.slider_int("Sides", ui_state.round_surface_sides, 3, 50)
                mesh_changed = mesh_changed or changed_rx or changed_rz or changed_se
            elif ui_state.add_mesh_type == AddType.Trapezoid:
                changed_tw, ui_state.trapezoid_top_width = imgui.drag_float("Top width", ui_state.trapezoid_top_width, 0.01, 0.01, 10.0)
                changed_bw, ui_state.trapezoid_bottom_width = imgui.drag_float("Bottom width", ui_state.trapezoid_bottom_width, 0.01, 0.01, 10.0)
                changed_h, ui_state.trapezoid_height = imgui.drag_float("Height", ui_state.trapezoid_height, 0.01, 0.01, 10.0)
                mesh_changed = mesh_changed or changed_tw or changed_bw or changed_h
            elif ui_state.add_mesh_type == AddType.Star:
                changed_ir, ui_state.star_inner_radius = imgui.drag_float("Inner radius", ui_state.star_inner_radius, 0.01, 0.01, 10.0)
                changed_or, ui_state.star_outer_radius = imgui.drag_float("Outer radius", ui_state.star_outer_radius, 0.01, 0.01, 10.0)
                changed_p, ui_state.star_points = imgui.slider_int("Points", ui_state.star_points, 2, 20)
                mesh_changed = mesh_changed or changed_ir or changed_or or changed_p
            elif ui_state.add_mesh_type == AddType.Arrow:
                changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
                changed_tl, ui_state.arrow_tail_length = imgui.drag_float("Tail length", ui_state.arrow_tail_length, 0.01, 0.01, 10.0)
                mesh_changed = mesh_changed or changed_s or changed_tl
            elif ui_state.add_mesh_type == AddType.Cube:
                changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
                mesh_changed = mesh_changed or changed_s
            elif ui_state.add_mesh_type == AddType.Tetrahedron:
                changed_s, ui_state.general_mesh_size = imgui.drag_float("Size", ui_state.general_mesh_size, 0.01, 0.01, 10.0)
                mesh_changed = mesh_changed or changed_s
            elif ui_state.add_mesh_type == AddType.Prism:
                changed_rb, ui_state.column_radius_bottom = imgui.drag_float("Radius", ui_state.column_radius_bottom, 0.01, 0.01, 10.0)
                changed_h, ui_state.column_height = imgui.drag_float("Height", ui_state.column_height, 0.01, 0.01, 10.0)
                changed_se, ui_state.column_sectors = imgui.slider_int("Sectors", ui_state.column_sectors, 3, 50)
                mesh_changed = mesh_changed or changed_rb or changed_h or changed_se
            elif ui_state.add_mesh_type == AddType.Cone:
                changed_rb, ui_state.column_radius_bottom = imgui.drag_float("Bottom radius", ui_state.column_radius_bottom, 0.01, 0.01, 10.0)
                changed_h, ui_state.column_height = imgui.drag_float("Height", ui_state.column_height, 0.01, 0.01, 10.0)
                changed_se, ui_state.column_sectors = imgui.slider_int("Sectors", ui_state.column_sectors, 3, 50)
                mesh_changed = mesh_changed or changed_rb or changed_h or changed_se
            elif ui_state.add_mesh_type == AddType.Cylinder:
                changed_rb, ui_state.column_radius_bottom = imgui.drag_float("Bottom radius", ui_state.column_radius_bottom, 0.01, 0.01, 10.0)
                changed_rt, ui_state.cylinder_radius_top = imgui.drag_float("Top radius", ui_state.cylinder_radius_top, 0.01, 0.01, 10.0)
                changed_h, ui_state.column_height = imgui.drag_float("Height", ui_state.column_height, 0.01, 0.01, 10.0)
                changed_se, ui_state.column_sectors = imgui.slider_int("Sectors", ui_state.column_sectors, 3, 50)
                mesh_changed = mesh_changed or changed_rb or changed_rt or changed_h or changed_se
            elif ui_state.add_mesh_type == AddType.UVSphere:
                changed_r, ui_state.sphere_radius = imgui.drag_float("Radius", ui_state.sphere_radius, 0.01, 0.01, 10.0)
                changed_st, ui_state.uv_sphere_stacks = imgui.slider_int("Stacks", ui_state.uv_sphere_stacks, 3, 50)
                changed_se, ui_state.uv_sphere_sectors = imgui.slider_int("Sectors", ui_state.uv_sphere_sectors, 3, 50)
                mesh_changed = mesh_changed or changed_r or changed_st or changed_se
            elif ui_state.add_mesh_type == AddType.Tetrasphere:
                changed_r, ui_state.sphere_radius = imgui.drag_float("Radius", ui_state.sphere_radius, 0.01, 0.01, 10.0)
                changed_sss, ui_state.subdiv_sphere_subdivisions = imgui.slider_int("Subdivisions", ui_state.subdiv_sphere_subdivisions, 1, 6)
                mesh_changed = mesh_changed or changed_r or changed_sss
            elif ui_state.add_mesh_type == AddType.Icosphere:
                changed_r, ui_state.sphere_radius = imgui.drag_float("Radius", ui_state.sphere_radius, 0.01, 0.01, 10.0)
                changed_sss, ui_state.subdiv_sphere_subdivisions = imgui.slider_int("Subdivisions", ui_state.subdiv_sphere_subdivisions, 1, 6)
                mesh_changed = mesh_changed or changed_r or changed_sss
            elif ui_state.add_mesh_type == AddType.Torus:
                changed_mr, ui_state.torus_main_radius = imgui.drag_float("Main radius", ui_state.torus_main_radius, 0.01, 0.01, 10.0)
                changed_tr, ui_state.torus_tube_radius = imgui.drag_float("Tube radius", ui_state.torus_tube_radius, 0.01, 0.01, 10.0)
                changed_mse, ui_state.torus_main_sectors = imgui.slider_int("Main sectors", ui_state.torus_main_sectors, 3, 50)
                changed_tse, ui_state.torus_tube_sectors = imgui.slider_int("Tube sectors", ui_state.torus_tube_sectors, 3, 50)
                mesh_changed = mesh_changed or changed_mr or changed_tr or changed_mse or changed_tse

            if ui_state.add_mesh_type in (AddType.DirectionalLight, AddType.PointLight, AddType.Camera, AddType.FunctionSurface, AddType.GradientDescentSurface):
                preview_visuals.enabled = False
            else:
                preview_visuals.enabled = True
                if mesh_changed:
                    vi = None
                    if ui_state.add_mesh_type == AddType.Triangle:
                        vi = generate_triangle(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Plane:
                        vi = generate_plane(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Pentagon:
                        vi = generate_pentagon(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Hexagon:
                        vi = generate_hexagon(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Circle:
                        vi = generate_circle(ui_state.general_mesh_size, ui_state.round_surface_sides)
                    elif ui_state.add_mesh_type == AddType.Ellipse:
                        vi = generate_ellipse(ui_state.ellipse_radius_x, ui_state.ellipse_radius_z, ui_state.round_surface_sides)
                    elif ui_state.add_mesh_type == AddType.Trapezoid:
                        vi = generate_trapezoid(ui_state.trapezoid_top_width, ui_state.trapezoid_bottom_width, ui_state.trapezoid_height)
                    elif ui_state.add_mesh_type == AddType.Star:
                        vi = generate_star(ui_state.star_outer_radius, ui_state.star_inner_radius, ui_state.star_points)
                    elif ui_state.add_mesh_type == AddType.Arrow:
                        vi = generate_arrow(ui_state.general_mesh_size, ui_state.arrow_tail_length)
                    elif ui_state.add_mesh_type == AddType.Cube:
                        vi = generate_cube(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Tetrahedron:
                        vi = generate_tetrahedron(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Prism:
                        vi = generate_prism(ui_state.column_sectors, ui_state.column_radius_bottom, ui_state.column_height)
                    elif ui_state.add_mesh_type == AddType.Cone:
                        vi = generate_cone(ui_state.column_radius_bottom, ui_state.column_height, ui_state.column_sectors)
                    elif ui_state.add_mesh_type == AddType.Cylinder:
                        vi = generate_cylinder(ui_state.column_radius_bottom, ui_state.cylinder_radius_top, ui_state.column_height, ui_state.column_sectors)
                    elif ui_state.add_mesh_type == AddType.UVSphere:
                        vi = generate_uv_sphere(ui_state.sphere_radius, ui_state.uv_sphere_stacks, ui_state.uv_sphere_sectors)
                    elif ui_state.add_mesh_type == AddType.Tetrasphere:
                        vi = generate_tetrasphere(ui_state.sphere_radius, ui_state.subdiv_sphere_subdivisions)
                    elif ui_state.add_mesh_type == AddType.Icosphere:
                        vi = generate_icosphere(ui_state.sphere_radius, ui_state.subdiv_sphere_subdivisions)
                    elif ui_state.add_mesh_type == AddType.Torus:
                        vi = generate_torus(ui_state.torus_main_radius, ui_state.torus_tube_radius, ui_state.torus_main_sectors, ui_state.torus_tube_sectors)

                    if vi is not None:
                        preview_visuals.mesh = AssetSystem.create_immediate_mesh(assets_state, *vi)
                    ui_state.preview_visual_initialized = True

            imgui.separator()
            if imgui.button("Add to scene"):
                new_entity = registry.create_entity()

                if ui_state.add_mesh_type in (
                    AddType.Triangle, AddType.Plane, AddType.Pentagon, AddType.Hexagon,
                    AddType.Circle, AddType.Ellipse, AddType.Trapezoid, AddType.Star, AddType.Arrow,
                    AddType.Cube, AddType.Tetrahedron,
                    AddType.Prism, AddType.Cone, AddType.Cylinder,
                    AddType.UVSphere, AddType.Tetrasphere, AddType.Icosphere, AddType.Torus
                ):
                    vi = None
                    if ui_state.add_mesh_type == AddType.Triangle:
                        vi = generate_triangle(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Plane:
                        vi = generate_plane(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Pentagon:
                        vi = generate_pentagon(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Hexagon:
                        vi = generate_hexagon(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Circle:
                        vi = generate_circle(ui_state.general_mesh_size, ui_state.round_surface_sides)
                    elif ui_state.add_mesh_type == AddType.Ellipse:
                        vi = generate_ellipse(ui_state.ellipse_radius_x, ui_state.ellipse_radius_z, ui_state.round_surface_sides)
                    elif ui_state.add_mesh_type == AddType.Trapezoid:
                        vi = generate_trapezoid(ui_state.trapezoid_top_width, ui_state.trapezoid_bottom_width, ui_state.trapezoid_height)
                    elif ui_state.add_mesh_type == AddType.Star:
                        vi = generate_star(ui_state.star_outer_radius, ui_state.star_inner_radius, ui_state.star_points)
                    elif ui_state.add_mesh_type == AddType.Arrow:
                        vi = generate_arrow(ui_state.general_mesh_size, ui_state.arrow_tail_length)
                    elif ui_state.add_mesh_type == AddType.Cube:
                        vi = generate_cube(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Tetrahedron:
                        vi = generate_tetrahedron(ui_state.general_mesh_size)
                    elif ui_state.add_mesh_type == AddType.Prism:
                        vi = generate_prism(ui_state.column_sectors, ui_state.column_radius_bottom, ui_state.column_height)
                    elif ui_state.add_mesh_type == AddType.Cone:
                        vi = generate_cone(ui_state.column_radius_bottom, ui_state.column_height, ui_state.column_sectors)
                    elif ui_state.add_mesh_type == AddType.Cylinder:
                        vi = generate_cylinder(ui_state.column_radius_bottom, ui_state.cylinder_radius_top, ui_state.column_height, ui_state.column_sectors)
                    elif ui_state.add_mesh_type == AddType.UVSphere:
                        vi = generate_uv_sphere(ui_state.sphere_radius, ui_state.uv_sphere_stacks, ui_state.uv_sphere_sectors)
                    elif ui_state.add_mesh_type == AddType.Tetrasphere:
                        vi = generate_tetrasphere(ui_state.sphere_radius, ui_state.subdiv_sphere_subdivisions)
                    elif ui_state.add_mesh_type == AddType.Icosphere:
                        vi = generate_icosphere(ui_state.sphere_radius, ui_state.subdiv_sphere_subdivisions)
                    elif ui_state.add_mesh_type == AddType.Torus:
                        vi = generate_torus(ui_state.torus_main_radius, ui_state.torus_tube_radius, ui_state.torus_main_sectors, ui_state.torus_tube_sectors)

                    if vi is not None:
                        new_material = copy.copy(ui_state.default_material)

                        registry.add_components(
                            new_entity,
                            EntityFlags(name=f"{ui_state.add_mesh_type.name}"),
                            dataclasses.replace(preview_transform),
                            Visuals(AssetSystem.create_immediate_mesh(assets_state, *vi), new_material)
                        )
                elif ui_state.add_mesh_type == AddType.FunctionSurface:
                    new_material = copy.copy(ui_state.default_material)
                    vi = generate_plane(10.0)

                    registry.add_components(
                        new_entity,
                        EntityFlags(name="Function surface"),
                        Transform(position=vec3(*camera_state.focal_point)),
                        Visuals(AssetSystem.create_immediate_mesh(assets_state, *vi), new_material, cull_back_faces=False),
                        SurfaceFunction()
                    )
                elif ui_state.add_mesh_type == AddType.GradientDescentSurface:
                    new_material = copy.copy(ui_state.default_material)
                    vi = generate_plane(10.0)

                    registry.add_components(
                        new_entity,
                        EntityFlags(name="Gradient descent surface"),

                        Transform(position=vec3(0.0, 0.0, 0.0), scale=vec3(1.0, 0.01, 1.0)),
                        GradientDescentSurface(),
                        Visuals(AssetSystem.create_immediate_mesh(assets_state, *vi), new_material, cull_back_faces=False),
                    )
                elif ui_state.add_mesh_type == AddType.DirectionalLight:
                    registry.add_components(
                        new_entity,
                        EntityFlags(name="Directional light"),
                        DirectionalLight()
                    )
                elif ui_state.add_mesh_type == AddType.PointLight:
                    registry.add_components(
                        new_entity,
                        EntityFlags(name="Point light"),
                        Transform(position=vec3(*camera_state.focal_point)),
                        PointLight()
                    )
                elif ui_state.add_mesh_type == AddType.Camera:
                    registry.add_components(
                        new_entity,
                        EntityFlags(name="Camera"),
                        Transform(position=vec3(*camera_state.focal_point)),
                        Camera()
                    )

                registry.set_parent(ui_state.selection_child_entity, new_entity)
                ui_state.should_close_add_menu = True

        # == entity list section ==
        if imgui.collapsing_header("Entities", imgui.TreeNodeFlags_.default_open):
            content_max_x = imgui.get_cursor_pos().x + imgui.get_content_region_avail().x
            ICON_PRIORITY = {
                Camera: icons_fontawesome_6.ICON_FA_CAMERA,
                DirectionalLight: icons_fontawesome_6.ICON_FA_SUN,
                PointLight: icons_fontawesome_6.ICON_FA_LIGHTBULB,
                Visuals: icons_fontawesome_6.ICON_FA_CUBE,
                Transform: icons_fontawesome_6.ICON_FA_ARROWS_UP_DOWN_LEFT_RIGHT,
            }
            FIXED_ICON_WIDTH = 26

            for entity_id, components in registry.view_all():
                entity_flags: EntityFlags | None = components.get(EntityFlags)
                if entity_flags is not None and entity_flags.is_internal:
                    continue

                display_name = entity_flags.name if (
                    entity_flags and entity_flags.name) else "Entity"

                display_icon = None
                for comp_type, icon in ICON_PRIORITY.items():
                    if comp_type in components:
                        display_icon = icon
                        break

                start_x = imgui.get_cursor_pos_x()

                if display_icon:
                    icon_width = imgui.calc_text_size(display_icon).x
                    center_offset = (FIXED_ICON_WIDTH - icon_width) / 2
                    imgui.set_cursor_pos_x(start_x + center_offset)
                    imgui.text_disabled(display_icon)
                    imgui.same_line(start_x + FIXED_ICON_WIDTH)
                else:
                    imgui.set_cursor_pos_x(start_x + FIXED_ICON_WIDTH)

                id_str = f"#{entity_id}"
                selectable_label = f"{display_name}###entity_{entity_id}"

                clicked, _ = imgui.selectable(selectable_label, selected_entity == entity_id)
                if clicked:
                    registry.set_parent(ui_state.selection_child_entity, entity_id)
                    selected_entity = entity_id  # is this necessary?

                id_width = imgui.calc_text_size(id_str).x
                align_x = content_max_x - id_width

                imgui.same_line(int(align_x))
                imgui.text_disabled(id_str)

        # == inspector section ==
        if selected_entity is None:
            header_title = f"Inspector###InspectorHeader"
            if imgui.collapsing_header(header_title, imgui.TreeNodeFlags_.default_open):
                imgui.text_disabled("Select an entity from the list to edit its properties.")
        else:
            selected_components = registry.get_all_components(selected_entity)

            entity_flags = selected_components.get(EntityFlags)
            if entity_flags is not None and entity_flags.name is not None:
                entity_label = f"{entity_flags.name} (#{selected_entity})"
            else:
                entity_label = f"Entity #{selected_entity}"

            # We use '###' so ImGui tracks the open/close state of the header even if the entity changes
            header_title = f"Inspector: {entity_label}###InspectorHeader"

            if imgui.collapsing_header(header_title, imgui.TreeNodeFlags_.default_open):
                if entity_flags is not None:
                    changed_name, new_name = imgui.input_text(
                        "Name", entity_flags.name if entity_flags.name is not None else "")
                    if changed_name:
                        entity_flags.name = new_name if new_name != "" else None
                    imgui.same_line()  # for the Delete button

                imgui.push_style_color(imgui.Col_.button, (0.8, 0.2, 0.2, 1.0))
                imgui.push_style_color(imgui.Col_.button_hovered, (0.9, 0.3, 0.3, 1.0))
                imgui.push_style_color(imgui.Col_.button_active, (1.0, 0.4, 0.4, 1.0))
                if imgui.button("Delete"):
                    disposal.entities_to_dispose.add(selected_entity)
                imgui.pop_style_color(3)

                if entity_flags is not None:
                    changed_classification, new_classification = imgui.drag_int(
                        "Classification", entity_flags.classification, 1, 0, 65536)
                    if changed_classification:
                        entity_flags.classification = new_classification

                target_camera = registry.get_parent(camera_state_entity)
                if target_camera is not None and target_camera != selected_entity:
                    cam_comps = registry.get_components(target_camera, Transform, Camera)
                    target_comps = registry.get_components(selected_entity, Transform)

                    if cam_comps is not None and target_comps is not None:
                        if imgui.button("Focus camera on this"):
                            (cam_transform, camera) = cam_comps
                            (target_transform, ) = target_comps

                            UiSystem.focus_camera_on_transform(
                                camera_state,
                                camera,
                                cam_transform,
                                target_transform
                            )

                imgui.separator()

                for comp_type, component in selected_components.items():
                    if comp_type is not EntityFlags:
                        UiSystem.draw_component_properties(
                            registry,
                            selected_entity, comp_type, component,
                            ui_state,
                            assets_state,
                            camera_state_entity, camera_state,
                        )

        # == graphics settings section ==
        if imgui.collapsing_header("Graphics", imgui.TreeNodeFlags_.default_open):
            if imgui.button("Capture this frame"):
                render_state.is_capture = True

            if imgui.begin_table("graphics_render_mode", 2):
                imgui.table_setup_column("Label", imgui.TableColumnFlags_.width_fixed)
                imgui.table_setup_column("Controls", imgui.TableColumnFlags_.width_stretch)

                imgui.table_next_row()

                imgui.table_next_column()
                imgui.align_text_to_frame_padding()
                imgui.text("Draw")

                imgui.table_next_column()
                if imgui.radio_button("normally", render_state.global_draw_mode == GlobalDrawMode.Normal):
                    render_state.global_draw_mode = GlobalDrawMode.Normal
                imgui.same_line()
                if imgui.radio_button("wireframe", render_state.global_draw_mode == GlobalDrawMode.Wireframe):
                    render_state.global_draw_mode = GlobalDrawMode.Wireframe
                imgui.same_line()
                if imgui.radio_button("depth", render_state.global_draw_mode == GlobalDrawMode.DepthOnly):
                    render_state.global_draw_mode = GlobalDrawMode.DepthOnly

                imgui.end_table()

            imgui.push_item_width(imgui.get_window_width() * 0.5)
            changed_li, new_li = imgui.checkbox("Light icons", icon_render_state.draw_light_icons)
            if changed_li:
                icon_render_state.draw_light_icons = new_li
            imgui.same_line()
            changed_ci, new_ci = imgui.checkbox("Camera icons", icon_render_state.draw_camera_icons)
            if changed_ci:
                icon_render_state.draw_camera_icons = new_ci

            if imgui.tree_node_ex("Shadow blurring"):
                changed_depth_sensitivity, new_depth_sensitivity = imgui.drag_float(
                    "Depth sensitivity", float(render_state.shadow_blur_depth_sensitivity), 0.01, 0.0, 200.0)
                if changed_depth_sensitivity:
                    render_state.shadow_blur_depth_sensitivity = float1(new_depth_sensitivity)
                changed_norm_thres, new_norm_thres = imgui.drag_float(
                    "Normal threshold", float(render_state.shadow_blur_normal_threshold), 0.001, 0.0, 1.0)
                if changed_norm_thres:
                    render_state.shadow_blur_normal_threshold = float1(new_norm_thres)

                imgui.tree_pop()

            for name, preset in [
                ("Viewport settings", render_state.viewport_graphics_settings),
                ("Capture settings", render_state.capture_graphics_settings)
            ]:
                if imgui.tree_node_ex(name):
                    changed_smaa, new_smaa = imgui.checkbox("SMAA", preset.enable_smaa)
                    if changed_smaa:
                        preset.enable_smaa = new_smaa

                    imgui.text_disabled("Shadows")

                    changed_p_pcf, new_p_pcf = imgui.drag_int(
                        "Point samples", preset.point_shadow_samples, 1, 1, 128)
                    if changed_p_pcf:
                        preset.point_shadow_samples = new_p_pcf

                    changed_d_pcf, new_d_pcf = imgui.drag_int(
                        "Directional samples", preset.directional_shadow_samples, 1, 1, 128)
                    if changed_d_pcf:
                        preset.directional_shadow_samples = new_d_pcf

                    imgui.tree_pop()

            imgui.pop_item_width()

        imgui.end()


    @staticmethod
    def draw_component_properties(
            registry: Registry,
            entity_id: int, comp_type: Type[Any], comp: Any,
            ui_state: UiState,
            assets_state: AssetsState,
            camera_state_entity: int, camera_state: CameraState,
    ):
        if isinstance(comp, Transform):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_pos, new_pos = imgui.drag_float3(
                    "Position", comp.position.tolist(), 0.1)
                if changed_pos:
                    comp.position = vec3(*new_pos)

                # no, we're not gonna be cute and use an epsilon or a dot product.
                # if it changed, it changed.
                if not np.array_equal(comp.rotation, ui_state.last_synced_quaternion):
                    new_euler = quaternion_to_euler(comp.rotation)
                    ui_state.euler_buffer = minimize_euler(new_euler)
                    ui_state.last_synced_quaternion = comp.rotation.copy()

                changed_rot, new_euler = imgui.drag_float3("Rotation", ui_state.euler_buffer.tolist(), 0.1)

                if changed_rot:
                    ui_state.euler_buffer = vec3(*new_euler)
                    new_quat_1, new_quat_2 = quaternions_from_euler(ui_state.euler_buffer)
                    if np.dot(comp.rotation, new_quat_1) >= 0:
                        comp.rotation = new_quat_1
                    else:
                        comp.rotation = new_quat_2
                    ui_state.last_synced_quaternion = comp.rotation.copy()

                changed_scale, new_scale = imgui.drag_float3(
                    "Scale", comp.scale.tolist(), 0.1)
                if changed_scale:
                    comp.scale = vec3(*new_scale)
                imgui.tree_pop()

        elif isinstance(comp, DirectionalLight):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_enabled, new_enabled = imgui.checkbox(
                    "Enabled", comp.enabled)
                if changed_enabled:
                    comp.enabled = new_enabled
                imgui.same_line()
                changed_casts_shadow, new_casts_shadow = imgui.checkbox(
                    "Casts shadow", comp.casts_shadow)
                if changed_casts_shadow:
                    comp.casts_shadow = new_casts_shadow
                # light color is stored as linear, but color_edit3 expects sRGB
                changed_color, new_color = imgui.color_edit3(
                    "Color", np.power(comp.color, 1.0/2.2).tolist())
                if changed_color:
                    comp.color = vec3(*np.power(new_color, 2.2))
                changed_strength, new_strength = imgui.drag_float(
                    "Strength", float(comp.strength), 1, 0.0, 1000.0)
                if changed_strength:
                    comp.strength = float1(new_strength)

                changed_rot, new_rot = imgui.drag_float3(
                    "Rotation", comp.rotation.tolist(), 0.1)
                if changed_rot:
                    comp.rotation = (vec3(*new_rot) + 180.0) % 360.0 - 180.0

                imgui.tree_pop()

        elif isinstance(comp, PointLight):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_enabled, new_enabled = imgui.checkbox(
                    "Enabled", comp.enabled)
                if changed_enabled:
                    comp.enabled = new_enabled
                imgui.same_line()
                changed_casts_shadow, new_casts_shadow = imgui.checkbox(
                    "Casts shadow", comp.casts_shadow)
                if changed_casts_shadow:
                    comp.casts_shadow = new_casts_shadow
                # light color is stored as linear, but color_edit3 expects sRGB
                changed_color, new_color = imgui.color_edit3(
                    "Color", np.power(comp.color, 1.0/2.2).tolist())
                if changed_color:
                    comp.color = vec3(*np.power(new_color, 2.2))
                changed_strength, new_strength = imgui.drag_float(
                    "Strength", float(comp.strength), 1, 0.0, 1000.0)
                if changed_strength:
                    comp.strength = float1(new_strength)
                imgui.tree_pop()

        elif isinstance(comp, Camera):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                is_targeted = registry.get_parent(camera_state_entity) == entity_id
                imgui.begin_disabled(is_targeted)
                changed_targeted, new_targeted = imgui.checkbox("Main camera", is_targeted)
                if changed_targeted:
                    registry.set_parent(camera_state_entity, entity_id if new_targeted else None)
                imgui.end_disabled()
                changed_fov, new_fov = imgui.drag_float("FOV", comp.fov, 1.0, 10.0, 150.0)
                if changed_fov:
                    comp.fov = new_fov
                changed_near, new_near = imgui.drag_float("Near plane", comp.near, 0.01)
                if changed_near:
                    comp.near = new_near
                changed_far, new_far = imgui.drag_float("Far plane", comp.far, 1.0)
                if changed_far:
                    comp.far = new_far
                imgui.tree_pop()

        elif isinstance(comp, Visuals) and not comp.is_internal:
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_enabled, new_enabled = imgui.checkbox(
                    "Shown", comp.enabled)
                if changed_enabled:
                    comp.enabled = new_enabled
                imgui.same_line()
                if imgui.radio_button("Fill", comp.draw_mode == DrawMode.Normal):
                    comp.draw_mode = DrawMode.Normal
                imgui.same_line()
                if imgui.radio_button("Wireframe", comp.draw_mode == DrawMode.Wireframe):
                    comp.draw_mode = DrawMode.Wireframe

                # albedo is stored as linear, but color_edit3 expects sRGB
                albedo_srgb = np.clip(comp.material.albedo ** (1.0 / 2.2), 0.0, 1.0)
                changed_albedo, new_albedo = imgui.color_edit3("Albedo", albedo_srgb.tolist())
                if changed_albedo:
                    comp.material.albedo = vec3(*np.power(new_albedo, 2.2))
                changed_roughness, new_roughness = imgui.slider_float("Roughness", float(comp.material.roughness), 0.0, 1.0)
                if changed_roughness:
                    comp.material.roughness = float1(new_roughness)
                changed_metallic, new_metallic = imgui.slider_float("Metallic", float(comp.material.metallic), 0.0, 1.0)
                if changed_metallic:
                    comp.material.metallic = float1(new_metallic)
                changed_reflectance, new_reflectance = imgui.slider_float("Reflectance", float(comp.material.reflectance), 0.0, 1.0)
                if changed_reflectance:
                    comp.material.reflectance = float1(new_reflectance)
                changed_translucency, new_translucency = imgui.slider_float("Translucency", float(comp.material.translucency), 0.0, 1.0)
                if changed_translucency:
                    comp.material.translucency = float1(new_translucency)
                changed_ao, new_ao = imgui.slider_float("AO", float(comp.material.ao), 0.0, 1.0)
                if changed_ao:
                    comp.material.ao = float1(new_ao)

                if imgui.tree_node_ex("Textures", imgui.TreeNodeFlags_.default_open):
                    if imgui.begin_table("material_textures_table", 3):
                        imgui.table_setup_column("Map", imgui.TableColumnFlags_.width_fixed)
                        imgui.table_setup_column("Status", imgui.TableColumnFlags_.width_fixed, 60.0)
                        imgui.table_setup_column("Controls", imgui.TableColumnFlags_.width_stretch)

                        texture_maps = [
                            ("Albedo", "albedo_map"),
                            # ("Normal", "normal_map"),
                            # ("Specular", "specular_map")
                        ]

                        for display_name, attr_name in texture_maps:
                            imgui.push_id(attr_name)
                            imgui.table_next_row()

                            imgui.table_next_column()
                            imgui.align_text_to_frame_padding()
                            imgui.text_unformatted(display_name)

                            imgui.table_next_column()
                            current_tex = getattr(comp.material, attr_name)

                            if current_tex is None:
                                imgui.align_text_to_frame_padding()
                                imgui.text_disabled("(None)")
                            elif current_tex.status == AssetStatus.Loading:
                                imgui.align_text_to_frame_padding()
                                imgui.text_colored((1.0, 0.8, 0.0, 1.0), "Loading...")
                            elif current_tex.status == AssetStatus.Failed:
                                imgui.align_text_to_frame_padding()
                                imgui.text_colored((1.0, 0.2, 0.2, 1.0), "Failed!")
                            elif current_tex.status == AssetStatus.Ready:
                                tex_ref = imgui.ImTextureRef(int(current_tex.gl_id))
                                imgui.image(tex_ref, imgui.ImVec2(24, 24), imgui.ImVec2(0, 1), imgui.ImVec2(1, 0))
                                if imgui.is_item_hovered():
                                    imgui.begin_tooltip()
                                    imgui.image(tex_ref, imgui.ImVec2(256, 256), imgui.ImVec2(0, 1), imgui.ImVec2(1, 0))
                                    imgui.end_tooltip()

                            imgui.table_next_column()
                            if imgui.button("Browse..."):
                                dialog = pfd.open_file(
                                    title=f"Select {display_name} Texture",
                                    default_path="",
                                    filters=["Image Files", "*.png *.jpg *.jpeg *.bmp *.tga", "All Files", "*"]
                                )
                                result = dialog.result()

                                if result and len(result) > 0:
                                    filepath = result[0]
                                    is_srgb = (attr_name == "albedo_map")
                                    new_tex = AssetSystem.request_texture(assets_state, filepath, is_srgb=is_srgb)
                                    setattr(comp.material, attr_name, new_tex)

                            if current_tex is not None:
                                imgui.same_line()
                                imgui.push_style_color(imgui.Col_.button, (0.6, 0.2, 0.2, 1.0))
                                if imgui.button("Clear"):
                                    setattr(comp.material, attr_name, None)
                                imgui.pop_style_color()

                            imgui.pop_id()

                        imgui.end_table()
                    imgui.tree_pop()

                imgui.tree_pop()

        elif isinstance(comp, SurfaceFunction):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_res, new_res = imgui.slider_int("Resolution", comp.resolution, 2, 200)
                if changed_res:
                    comp.resolution = new_res
                    comp.generated = False

                changed_size, new_size = imgui.drag_float("Size", comp.size, 0.1, 0.1, 100.0)
                if changed_size:
                    comp.size = new_size
                    comp.generated = False

                imgui.separator()

                label = "*Expression" if comp.expression_dirty else "Expression"
                changed_expr, new_expr = imgui.input_text(f"{label}###expr_input", comp.expression)

                if changed_expr:
                    comp.expression = new_expr
                    comp.expression_dirty = True

                if imgui.button("Compile"):
                    comp.expression_dirty = False
                    comp.generated = False

                imgui.push_text_wrap_pos(0.0)
                if comp.error_status == CompilationStatus.Ok:
                    imgui.text_colored((0.2, 0.8, 0.2, 1.0), comp.error_string)
                elif comp.error_status == CompilationStatus.Warning:
                    imgui.text_colored(
                        (1.0, 0.8, 0.0, 1.0), f"{icons_fontawesome_6.ICON_FA_TRIANGLE_EXCLAMATION} {comp.error_string}")
                elif comp.error_status == CompilationStatus.Error:
                    imgui.text_colored(
                        (0.9, 0.2, 0.2, 1.0), f"{icons_fontawesome_6.ICON_FA_CIRCLE_EXCLAMATION} {comp.error_string}")
                imgui.pop_text_wrap_pos()

                imgui.tree_pop()

        elif isinstance(comp, GradientDescentSurface):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_run, new_run = imgui.checkbox("Running", comp.is_running)
                if changed_run:
                    comp.is_running = new_run

                changed_interval, new_interval = imgui.slider_float("Step interval", comp.step_interval, 0.005, 0.25)
                if changed_interval:
                    comp.step_interval = new_interval

                imgui.separator()

                func_names = [e.name for e in LossFunctionType]
                current_index = list(LossFunctionType).index(comp.function_type)
                changed_func, new_index = imgui.combo("Function", current_index, func_names)
                if changed_func:
                    comp.function_type = list(LossFunctionType)[new_index]
                    comp.dirty = True

                if comp.function_type == LossFunctionType.Rosenbrock:
                    changed_a, new_a = imgui.drag_float("a", comp.rosenbrock_a, 0.1)
                    if changed_a:
                        comp.rosenbrock_a = new_a
                        comp.dirty = True
                    changed_b, new_b = imgui.drag_float("b", comp.rosenbrock_b, 1.0)
                    if changed_b:
                        comp.rosenbrock_b = new_b
                        comp.dirty = True

                changed_res, new_res = imgui.slider_int("Resolution", comp.resolution, 10, 200)
                if changed_res:
                    comp.resolution = new_res
                    comp.dirty = True

                changed_size, new_size = imgui.drag_float("Size", comp.size, 0.1, 1.0, 100.0)
                if changed_size:
                    comp.size = new_size
                    comp.dirty = True

                imgui.text_disabled(f"Iterations: {comp.iterations}")

                if imgui.button("+ Add optimizer"):
                    r_transform = registry.get_components(entity_id, Transform)
                    spawn_pos = vec3(0.0, 0.0, 0.0)
                    if r_transform:
                        (surf_trans,) = r_transform
                        spawn_pos = surf_trans.position.copy()

                    optimizer_mesh = AssetSystem.create_immediate_mesh(assets_state, *generate_uv_sphere(radius=0.1, stacks=10, sectors=20))

                    new_entity = registry.create_entity()
                    registry.add_components(
                        new_entity,
                        EntityFlags(name=f"Optimizer {new_entity}"),
                        Transform(position=spawn_pos),
                        OptimizerState(algorithm=OptimizerAlgorithm.BatchGD),
                        Visuals(optimizer_mesh, copy.copy(ui_state.default_material)),
                    )

                    registry.set_parent(new_entity, entity_id)
                    registry.set_parent(ui_state.selection_child_entity, new_entity)

                imgui.tree_pop()

        elif isinstance(comp, OptimizerState):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                alg_names = [e.name for e in OptimizerAlgorithm]
                current_index = list(OptimizerAlgorithm).index(comp.algorithm)
                changed_alg, new_index = imgui.combo("Algorithm", current_index, alg_names)
                if changed_alg:
                    comp.algorithm = list(OptimizerAlgorithm)[new_index]

                changed_lr, new_lr = imgui.drag_float(
                    "Learning rate", comp.learning_rate, 0.0001, 0.0001, 1.0, "%.4f")
                if changed_lr:
                    comp.learning_rate = new_lr

                if comp.algorithm == OptimizerAlgorithm.Momentum:
                    changed_mom, new_mom = imgui.slider_float("Momentum Rate", comp.momentum_rate, 0.0, 1.0)
                    if changed_mom:
                        comp.momentum_rate = new_mom

                if comp.algorithm in (OptimizerAlgorithm.SGD, OptimizerAlgorithm.MiniBatchSGD):
                    changed_noise, new_noise = imgui.drag_float("Noise Scale", comp.noise_scale, 0.1, 0.0, 100.0)
                    if changed_noise:
                        comp.noise_scale = new_noise

                imgui.separator()

                if imgui.button("Clear trajectory & velocity"):
                    comp.trajectory.clear()
                    comp.velocity_x = 0.0
                    comp.velocity_z = 0.0

                imgui.text_disabled(f"Velocity: ({comp.velocity_x:.4f}, {comp.velocity_z:.4f})")
                imgui.tree_pop()

        else:
            UiSystem.disabled_bullet(comp_type.__name__)

    @staticmethod
    def disabled_bullet(text):
        imgui.push_style_var(imgui.StyleVar_.alpha,
                             imgui.get_style().alpha * 0.5)
        disabled_color = imgui.get_style().color_(imgui.Col_.text_disabled)
        imgui.push_style_color(imgui.Col_.text, disabled_color)
        imgui.bullet_text(text)
        imgui.pop_style_color(1)
        imgui.pop_style_var(1)

    @staticmethod
    def focus_camera_on_transform(
        camera_state: CameraState,
        camera: Camera,
        camera_transform: Transform,
        target_transform: Transform,
    ):
        camera_state.focal_point = target_transform.position
        camera_transform.position = camera_state.focal_point - camera_state.front * camera.focal_point_distance  # type: ignore
