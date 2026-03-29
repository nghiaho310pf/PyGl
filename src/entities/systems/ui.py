import copy
import dataclasses
import math
from typing import Any, Type

import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6

from entities.components.camera import Camera
from entities.components.camera_state import CameraState
from entities.components.icon_render_state import IconRenderState
from entities.components.point_light import PointLight
from entities.components.render_state import GlobalDrawMode, RenderState
from entities.components.surface_function import CompilationStatus, SurfaceFunction
from entities.components.transform import Transform
from entities.components.ui_state import UiState, AddType
from entities.components.visuals import DrawMode, Visuals
from entities.components.entity_flags import EntityFlags
from entities.registry import Registry
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
from meshes.mesh import Mesh
from math_utils import vec3
from shading.material import Material, ShaderType


class UiSystem:
    @staticmethod
    def update(registry: Registry, time: float, delta_time: float):
        r_admin = registry.get_singleton(UiState, CameraState, RenderState, IconRenderState)
        if r_admin is None:
            return
        admin_entity, (ui_state, camera_state, render_state, icon_render_state) = r_admin
        r_preview = registry.get_components(ui_state.preview_entity, Transform, Visuals)
        if r_preview is None:
            return
        (preview_transform, preview_visuals) = r_preview

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

        # == creation section ==
        if ui_state.should_close_add_menu:
            imgui.set_next_item_open(False)
            ui_state.should_close_add_menu = False
        add_menu_expanded = imgui.collapsing_header("Add")

        if not add_menu_expanded:
            preview_visuals.enabled = False
        else:
            changed_pos, new_pos = imgui.drag_float3(
                "Position", preview_transform.position.tolist(), 0.1)
            if changed_pos:
                preview_transform.position = vec3(*new_pos)

            changed_rot, new_rot = imgui.drag_float3(
                "Rotation", preview_transform.rotation.tolist(), 1.0)
            if changed_rot:
                preview_transform.rotation = vec3(*new_rot)

            changed_scale, new_scale = imgui.drag_float3(
                "Scale", preview_transform.scale.tolist(), 0.1)
            if changed_scale:
                preview_transform.scale = vec3(*new_scale)

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

            if imgui.radio_button("Light", ui_state.add_mesh_type == AddType.PointLight):
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

            if ui_state.add_mesh_type in (AddType.PointLight, AddType.Camera, AddType.FunctionSurface):
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
                        preview_visuals.mesh = Mesh(*vi)
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
                        new_material = Material(
                            ui_state.default_material.shader_type,
                            copy.deepcopy(ui_state.default_material.properties)
                        )

                        registry.add_components(
                            new_entity,
                            EntityFlags(name=f"{ui_state.add_mesh_type.name}"),
                            dataclasses.replace(preview_transform),
                            Visuals(Mesh(*vi), new_material)
                        )
                elif ui_state.add_mesh_type == AddType.FunctionSurface:
                    new_material = Material(
                        ui_state.default_material.shader_type,
                        copy.deepcopy(ui_state.default_material.properties)
                    )

                    # We create an initial plane mesh as a placeholder until the System ticks
                    vi = generate_plane(10.0)

                    registry.add_components(
                        new_entity,
                        EntityFlags(name="Function surface"),
                        Transform(position=vec3(0.0, 0.0, 0.0)),
                        Visuals(Mesh(*vi), new_material, cull_back_faces=False),
                        SurfaceFunction() # Our new component
                    )
                elif ui_state.add_mesh_type == AddType.PointLight:
                    registry.add_components(
                        new_entity,
                        EntityFlags(name="Point light"),
                        Transform(position=vec3(0.0, 0.0, 0.0)),
                        PointLight()
                    )
                elif ui_state.add_mesh_type == AddType.Camera:
                    registry.add_components(
                        new_entity,
                        EntityFlags(name="Camera"),
                        Transform(position=vec3(0.0, 0.0, 0.0)),
                        Camera()
                    )

                ui_state.selected_entity = new_entity
                ui_state.should_close_add_menu = True

        # == entity list section ==
        if imgui.collapsing_header("Entities", imgui.TreeNodeFlags_.default_open):
            content_max_x = imgui.get_cursor_pos().x + imgui.get_content_region_avail().x
            ICON_PRIORITY = {
                Camera: icons_fontawesome_6.ICON_FA_CAMERA,
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

                clicked, _ = imgui.selectable(
                    selectable_label, ui_state.selected_entity == entity_id)
                if clicked:
                    ui_state.selected_entity = entity_id

                id_width = imgui.calc_text_size(id_str).x
                align_x = content_max_x - id_width

                imgui.same_line(int(align_x))
                imgui.text_disabled(id_str)

        # == inspector section ==
        if ui_state.selected_entity is not None:
            selected_components = registry.get_all_components(
                ui_state.selected_entity)

            entity_flags = selected_components.get(EntityFlags)
            if entity_flags is not None and entity_flags.name is not None:
                entity_label = f"{entity_flags.name} (#{ui_state.selected_entity})"
            else:
                entity_label = f"Entity #{ui_state.selected_entity}"

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
                    ui_state.entities_to_dispose.append(ui_state.selected_entity)
                imgui.pop_style_color(3)

                target_camera = camera_state.target_camera
                if target_camera is not None and target_camera != ui_state.selected_entity:
                    cam_comps = registry.get_components(target_camera, Transform, Camera)
                    target_comps = registry.get_components(ui_state.selected_entity, Transform)

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
                    if comp_type != EntityFlags:
                        UiSystem.draw_component_properties(
                            ui_state.selected_entity, comp_type, component,
                            camera_state,
                        )

        # == debug section ==
        if imgui.collapsing_header("Debug", imgui.TreeNodeFlags_.default_open):
            if imgui.begin_table("debug_visualize_icons", 2):
                imgui.table_setup_column("Label", imgui.TableColumnFlags_.width_fixed)
                imgui.table_setup_column("Controls", imgui.TableColumnFlags_.width_stretch)

                imgui.table_next_row()
                
                imgui.table_next_column()
                imgui.align_text_to_frame_padding()
                imgui.text("Visualize")

                imgui.table_next_column()
                changed_li, new_li = imgui.checkbox("lights", icon_render_state.draw_light_icons)
                if changed_li:
                    icon_render_state.draw_light_icons = new_li
                imgui.same_line()
                changed_ci, new_ci = imgui.checkbox("cameras", icon_render_state.draw_camera_icons)
                if changed_ci:
                    icon_render_state.draw_camera_icons = new_ci

                imgui.end_table()

            if imgui.begin_table("debug_render_mode", 2):
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
                if imgui.radio_button("depthmap", render_state.global_draw_mode == GlobalDrawMode.DepthOnly):
                    render_state.global_draw_mode = GlobalDrawMode.DepthOnly

                imgui.end_table()

        imgui.end()

        for e in ui_state.entities_to_dispose:
            if e == ui_state.selected_entity:
                ui_state.selected_entity = None
            registry.remove_entity(e)

    @staticmethod
    def draw_component_properties(
            entity_id: int, comp_type: Type[Any], comp: Any,
            camera_state: CameraState,
    ):
        if isinstance(comp, Transform):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_pos, new_pos = imgui.drag_float3(
                    "Position", comp.position.tolist(), 0.1)
                if changed_pos:
                    comp.position = vec3(*new_pos)

                changed_rot, new_rot = imgui.drag_float3(
                    "Rotation", comp.rotation.tolist(), 1.0)
                if changed_rot:
                    comp.rotation = vec3(*new_rot)

                changed_scale, new_scale = imgui.drag_float3(
                    "Scale", comp.scale.tolist(), 0.1)
                if changed_scale:
                    comp.scale = vec3(*new_scale)
                imgui.tree_pop()

        elif isinstance(comp, PointLight):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_enabled, new_enabled = imgui.checkbox(
                    "Enabled", comp.enabled)
                if changed_enabled:
                    comp.enabled = new_enabled
                changed_color, new_color = imgui.color_edit3(
                    "Color", comp.color.tolist())
                if changed_color:
                    comp.color = vec3(*new_color)
                changed_strength, new_strength = imgui.drag_float(
                    "Strength", float(comp.strength), 1, 0.0, 1000.0)
                if changed_strength:
                    comp.strength = np.float32(new_strength)
                imgui.tree_pop()

        elif isinstance(comp, Camera):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                is_targeted = camera_state.target_camera == entity_id
                imgui.begin_disabled(is_targeted)
                changed_targeted, new_targeted = imgui.checkbox("Main camera", is_targeted)
                if changed_targeted:
                    camera_state.target_camera = entity_id
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

                if imgui.radio_button("Flat", comp.material.shader_type == ShaderType.Flat):
                    comp.material.shader_type = ShaderType.Flat
                imgui.same_line()
                if imgui.radio_button("Blinn-Phong", comp.material.shader_type == ShaderType.BlinnPhong):
                    comp.material.shader_type = ShaderType.BlinnPhong
                imgui.same_line()
                if imgui.radio_button("Gouraud", comp.material.shader_type == ShaderType.Gouraud):
                    comp.material.shader_type = ShaderType.Gouraud

                # for now just a hardcoded whitelist of the parameters
                albedo = comp.material.properties["u_Albedo"]
                if albedo is not None:
                    changed_albedo, new_albedo = imgui.color_edit3(
                        "Albedo", albedo)
                    if changed_albedo:
                        comp.material.properties["u_Albedo"] = new_albedo

                roughness = comp.material.properties["u_Roughness"]
                if roughness is not None:
                    changed_roughness, new_roughness = imgui.slider_float(
                        "Roughness", roughness, 0.0, 1.0)
                    if changed_roughness:
                        comp.material.properties["u_Roughness"] = new_roughness

                reflectance = comp.material.properties["u_Reflectance"]
                if reflectance is not None:
                    changed_reflectance, new_reflectance = imgui.slider_float(
                        "Reflectance", reflectance, 0.0, 1.0)
                    if changed_reflectance:
                        comp.material.properties["u_Reflectance"] = new_reflectance

                ao = comp.material.properties["u_AO"]
                if ao is not None:
                    changed_ao, new_ao = imgui.slider_float("AO", ao, 0.0, 1.0)
                    if changed_ao:
                        comp.material.properties["u_AO"] = new_ao

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

                if comp.error_status == CompilationStatus.Ok:
                    imgui.text_colored((0.2, 0.8, 0.2, 1.0), comp.error_string)
                elif comp.error_status == CompilationStatus.Warning:
                    imgui.text_colored((0.8, 0.2, 0.2, 1.0), comp.error_string)
                elif comp.error_status == CompilationStatus.Error:
                    imgui.text_colored((0.9, 0.2, 0.2, 1.0), comp.error_string)

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
        camera_state.focal_point = np.array(target_transform.position, dtype=np.float32)

        pitch_rad, yaw_rad, roll_rad = np.radians(camera_transform.rotation)
        front = np.array([
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad)
        ], dtype=np.float32)

        norm = np.linalg.norm(front)
        if norm > 0:
            front /= norm

        cam_pos = camera_state.focal_point - front * camera.focal_point_distance
        camera_transform.position = vec3(*cam_pos)