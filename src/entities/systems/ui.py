from typing import Any, Type

import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6, portable_file_dialogs as pfd

from entities.components.camera import Camera
from entities.components.camera_state import CameraState
from entities.components.directional_light import DirectionalLight
from entities.components.disposal import Disposal
from entities.components.ui.icon_render_state import IconRenderState
from entities.components.point_light import PointLight
from entities.components.render_state import GlobalDrawMode, RenderState
from entities.components.transform import Transform
from entities.components.ui.ui_state import UiState
from entities.components.visuals.visuals import DrawMode, Visuals
from entities.components.entity_flags import EntityFlags
from entities.components.visuals.assets import AssetStatus, AssetsState
from entities.systems.assets import AssetSystem
from entities.registry import Registry
from math_utils import float1, vec3


class UiSystem:
    @staticmethod
    def update(registry: Registry, time: float, delta_time: float):
        r_admin = registry.get_singleton(UiState, RenderState, IconRenderState, Disposal)
        r_camera_state = registry.get_singleton(CameraState)
        if r_admin is None or r_camera_state is None:
            return
        admin_entity, (ui_state, render_state, icon_render_state, disposal) = r_admin
        camera_state_entity, (camera_state, ) = r_camera_state

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

        # warn when there's no camera
        if registry.get_parent(camera_state_entity) is None:
            imgui.text_colored((1.0, 0.8, 0.0, 1.0), f"{icons_fontawesome_6.ICON_FA_TRIANGLE_EXCLAMATION} No camera.")

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
        if selected_entity is not None:
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
                            camera_state_entity, camera_state,
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


    @staticmethod
    def draw_component_properties(
            registry: Registry,
            entity_id: int, comp_type: Type[Any], comp: Any,
            ui_state: UiState,
            camera_state_entity: int, camera_state: CameraState,
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
                    comp.rotation = (vec3(*new_rot) + 180.0) % 360.0 - 180.0

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
                changed_color, new_color = imgui.color_edit3(
                    "Color", comp.color.tolist())
                if changed_color:
                    comp.color = vec3(*new_color)
                changed_strength, new_strength = imgui.drag_float(
                    "Strength", float(comp.strength), 1, 0.0, 1000.0)
                if changed_strength:
                    comp.strength = float1(new_strength)

                changed_rot, new_rot = imgui.drag_float3(
                    "Rotation", comp.rotation.tolist(), 1.0)
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
                changed_color, new_color = imgui.color_edit3(
                    "Color", comp.color.tolist())
                if changed_color:
                    comp.color = vec3(*new_color)
                changed_strength, new_strength = imgui.drag_float(
                    "Strength", float(comp.strength), 1, 0.0, 1000.0)
                if changed_strength:
                    comp.strength = float1(new_strength)
                changed_radius, new_radius = imgui.drag_float(
                    "Radius", float(comp.radius), 0.001, 0.0, 0.1)
                if changed_radius:
                    comp.radius = float1(new_radius)
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

                changed_albedo, new_albedo = imgui.color_edit3("Albedo", comp.material.albedo.tolist())
                if changed_albedo:
                    comp.material.albedo = vec3(*new_albedo)                    
                changed_roughness, new_roughness = imgui.slider_float("Roughness", float(comp.material.roughness), 0.0, 1.0)
                if changed_roughness:
                    comp.material.roughness = float1(new_roughness)
                changed_reflectance, new_reflectance = imgui.slider_float("Reflectance", float(comp.material.reflectance), 0.0, 1.0)
                if changed_reflectance:
                    comp.material.reflectance = float1(new_reflectance)
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

                                    r_assets = registry.get_singleton(AssetsState)
                                    if r_assets:
                                        _, (textures_state, ) = r_assets
                                        new_tex = AssetSystem.request_texture(textures_state, filepath)
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
        cam_pos = camera_state.focal_point - camera_state.front * camera.focal_point_distance
        camera_transform.position = vec3(*cam_pos)
