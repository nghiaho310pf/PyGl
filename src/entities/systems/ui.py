import copy
from typing import Any, Type

import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6

from entities.components.camera import Camera
from entities.components.point_light import PointLight
from entities.components.render_state import DrawMode, RenderState
from entities.components.transform import Transform
from entities.components.ui_state import UiState, AddType
from entities.components.visuals import Visuals
from entities.components.entity_flags import EntityFlags
from entities.registry import Registry
from geometry.geometry import generate_cube_flat, generate_plane, generate_sphere
from geometry.mesh import Mesh
from math_utils import vec3
from shading.material import Material


class UiSystem:
    @staticmethod
    def update(registry: Registry, time: float, delta_time: float):
        r_ui = registry.get_singleton(UiState, Visuals)
        r_render = registry.get_singleton(RenderState)
        if r_ui is None or r_render is None:
            return
        ui_state_entity, (ui_state, ui_visuals) = r_ui
        render_state_entity, (render_state, ) = r_render

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
            ui_visuals.enabled = False
        else:
            changed_type = False

            if imgui.radio_button("Sphere", ui_state.add_mesh_type == AddType.Sphere):
                ui_state.add_mesh_type = AddType.Sphere
                changed_type = True
            if imgui.radio_button("Cube", ui_state.add_mesh_type == AddType.Cube):
                ui_state.add_mesh_type = AddType.Cube
                changed_type = True
            if imgui.radio_button("Plane", ui_state.add_mesh_type == AddType.Plane):
                ui_state.add_mesh_type = AddType.Plane
                changed_type = True
            if imgui.radio_button("Light", ui_state.add_mesh_type == AddType.PointLight):
                ui_state.add_mesh_type = AddType.PointLight
                changed_type = True
            if imgui.radio_button("Camera", ui_state.add_mesh_type == AddType.Camera):
                ui_state.add_mesh_type = AddType.Camera
                changed_type = True

            mesh_changed = not ui_state.preview_visual_initialized or changed_type

            if ui_state.add_mesh_type == AddType.Sphere:
                changed_r, ui_state.sphere_radius = imgui.slider_float("Radius", ui_state.sphere_radius, 0.1, 5.0)
                changed_st, ui_state.sphere_stacks = imgui.slider_int("Stacks", ui_state.sphere_stacks, 3, 50)
                changed_se, ui_state.sphere_sectors = imgui.slider_int("Sectors", ui_state.sphere_sectors, 3, 50)
                mesh_changed = mesh_changed or changed_r or changed_st or changed_se

            elif ui_state.add_mesh_type == AddType.Cube:
                changed_s, ui_state.cube_size = imgui.slider_float("Size", ui_state.cube_size, 0.1, 5.0)
                mesh_changed = mesh_changed or changed_s

            if ui_state.add_mesh_type in (AddType.PointLight, AddType.Camera):
                ui_visuals.enabled = False
            else:
                ui_visuals.enabled = True
                if mesh_changed:
                    vi = None
                    if ui_state.add_mesh_type == AddType.Sphere:
                        vi = generate_sphere(ui_state.sphere_radius, ui_state.sphere_stacks, ui_state.sphere_sectors)
                    elif ui_state.add_mesh_type == AddType.Cube:
                        vi = generate_cube_flat(ui_state.cube_size)
                    elif ui_state.add_mesh_type == AddType.Plane:
                        vi = generate_plane()

                    if vi is not None:
                        ui_visuals.mesh = Mesh(*vi)
                    ui_state.preview_visual_initialized = True

            imgui.separator()
            if imgui.button("Add to scene"):
                new_entity = registry.create_entity()

                if ui_state.add_mesh_type in (AddType.Sphere, AddType.Cube, AddType.Plane):
                    vi = None
                    if ui_state.add_mesh_type == AddType.Sphere:
                        vi = generate_sphere(ui_state.sphere_radius, ui_state.sphere_stacks, ui_state.sphere_sectors)
                    elif ui_state.add_mesh_type == AddType.Cube:
                        vi = generate_cube_flat(ui_state.cube_size)
                    elif ui_state.add_mesh_type == AddType.Plane:
                        vi = generate_plane()

                    if vi is not None:
                        new_material = Material(
                            ui_state.default_material.shader,
                            copy.deepcopy(ui_state.default_material.properties)
                        )

                        registry.add_components(
                            new_entity,
                            EntityFlags(name=f"{ui_state.add_mesh_type.name}"),
                            Transform(position=vec3(0.0, 0.0, 0.0)),
                            Visuals(Mesh(*vi), new_material)
                        )
                elif ui_state.add_mesh_type == AddType.PointLight:
                    registry.add_components(
                        new_entity,
                        EntityFlags(name="Point Light"),
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
                for comp_type, component in selected_components.items():
                    UiSystem.draw_component_properties(
                        ui_state, render_state,
                        ui_state.selected_entity, comp_type, component
                    )

        # == debug section ==
        if imgui.collapsing_header("Debug", imgui.TreeNodeFlags_.default_open):
            if imgui.radio_button("Draw normally", render_state.draw_mode == DrawMode.Normal):
                render_state.draw_mode = DrawMode.Normal
            if imgui.radio_button("Draw wireframe", render_state.draw_mode == DrawMode.Wireframe):
                render_state.draw_mode = DrawMode.Wireframe
            if imgui.radio_button("Draw depth", render_state.draw_mode == DrawMode.DepthOnly):
                render_state.draw_mode = DrawMode.DepthOnly

        imgui.end()

        for e in ui_state.entities_to_dispose:
            if e == ui_state.selected_entity:
                ui_state.selected_entity = None
            registry.remove_entity(e)

    @staticmethod
    def draw_component_properties(
            ui_state: UiState, render_state: RenderState,
            entity_id: int, comp_type: Type[Any], comp: Any
    ):
        if isinstance(comp, EntityFlags):
            changed_name, new_name = imgui.input_text(
                "Name", comp.name if comp.name is not None else "")
            if changed_name:
                comp.name = new_name if new_name != "" else None
            
            imgui.spacing()

            imgui.push_style_color(imgui.Col_.button, (0.8, 0.2, 0.2, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, (0.9, 0.3, 0.3, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, (1.0, 0.4, 0.4, 1.0))            
            if imgui.button("Delete"):
                ui_state.entities_to_dispose.append(entity_id)
            imgui.pop_style_color(3)
            imgui.separator()

        elif isinstance(comp, Transform):
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
                is_targeted = render_state.target_camera == entity_id
                imgui.begin_disabled(is_targeted)
                changed_targeted, new_targeted = imgui.checkbox("Main camera", is_targeted)
                if changed_targeted:
                    render_state.target_camera = entity_id
                imgui.end_disabled()

                changed_fov, new_fov = imgui.drag_float(
                    "FOV", comp.fov, 1.0, 10.0, 150.0)
                if changed_fov:
                    comp.fov = new_fov

                changed_near, new_near = imgui.drag_float(
                    "Near Plane", comp.near, 0.01)
                if changed_near:
                    comp.near = new_near

                changed_far, new_far = imgui.drag_float(
                    "Far Plane", comp.far, 1.0)
                if changed_far:
                    comp.far = new_far
                imgui.tree_pop()

        elif isinstance(comp, Visuals) and not comp.is_internal:
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_enabled, new_enabled = imgui.checkbox(
                    "Shown", comp.enabled)
                if changed_enabled:
                    comp.enabled = new_enabled

                imgui.tree_pop()

            # hardcode "Material", there might be a separate mesh-related dropdown later
            if imgui.tree_node_ex("Material", imgui.TreeNodeFlags_.default_open):
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
