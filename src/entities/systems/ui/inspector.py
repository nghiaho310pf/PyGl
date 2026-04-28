import copy
from typing import Any, Type
import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6, portable_file_dialogs as pfd

from entities.components.camera import Camera
from entities.components.camera_state import CameraState
from entities.components.street_scene.scene_generator_state import SceneGeneratorState
from entities.components.street_scene.scene_animator_state import SceneAnimatorState
from entities.components.directional_light import DirectionalLight
from entities.components.disposal import Disposal
from entities.components.gd.optimizer_state import OptimizerAlgorithm, OptimizerState
from entities.components.gd.surface import GradientDescentSurface, LossFunctionType
from entities.components.surface_function import CompilationStatus, SurfaceFunction
from entities.components.point_light import PointLight
from entities.components.transform import Transform, TransformData
from entities.components.ui.ui_state import UiState
from entities.components.visuals.visuals import DrawMode, Visuals
from entities.components.entity_flags import EntityFlags
from entities.components.visuals.assets import AssetStatus, AssetsState
from entities.systems.assets import AssetSystem
from entities.registry import Registry
from meshes.volumes.uv_sphere import generate_uv_sphere
from math_utils import float1, minimize_euler, quaternions_from_euler, vec3, quaternion_to_euler


def draw_inspector_section(
    registry: Registry,
    selected_entity: int | None,
    ui_state: UiState,
    assets_state: AssetsState,
    camera_state_entity: int,
    camera_state: CameraState,
    disposal: Disposal,
):
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

                        focus_camera_on_transform(
                            camera_state,
                            camera,
                            cam_transform,
                            target_transform
                        )

            imgui.separator()

            for comp_type, component in selected_components.items():
                if comp_type is not EntityFlags:
                    draw_component_properties(
                        registry,
                        selected_entity, comp_type, component,
                        ui_state,
                        assets_state,
                        camera_state_entity, camera_state,
                    )


def draw_component_properties(
    registry: Registry,
    entity_id: int, comp_type: Type[Any], comp: Any,
    ui_state: UiState,
    assets_state: AssetsState,
    camera_state_entity: int, camera_state: CameraState,
):
    if isinstance(comp, Transform):
        if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
            changed_inherit, new_inherit = imgui.checkbox("Inherit", comp.inherit)
            if changed_inherit:
                comp.inherit = new_inherit

            imgui.same_line()
            imgui.text_disabled(f"World position: ({comp.world.position[0]:.3f}, {comp.world.position[1]:.3f}, {comp.world.position[2]:.3f})")

            changed_pos, new_pos = imgui.drag_float3(
                "Position", comp.local.position.tolist(), 0.1)
            if changed_pos:
                comp.local.position = vec3(*new_pos)

            # no, we're not gonna be cute and use an epsilon or a dot product.
            # if it changed, it changed.
            if not np.array_equal(comp.local.rotation, ui_state.last_synced_quaternion):
                new_euler = quaternion_to_euler(comp.local.rotation)
                ui_state.euler_buffer = minimize_euler(new_euler)
                ui_state.last_synced_quaternion = comp.local.rotation.copy()

            changed_rot, new_euler = imgui.drag_float3("Rotation", ui_state.euler_buffer.tolist(), 0.1)

            if changed_rot:
                ui_state.euler_buffer = vec3(*new_euler)
                new_quat_1, new_quat_2 = quaternions_from_euler(ui_state.euler_buffer)
                if np.dot(comp.local.rotation, new_quat_1) >= 0:
                    comp.local.rotation = new_quat_1
                else:
                    comp.local.rotation = new_quat_2
                ui_state.last_synced_quaternion = comp.local.rotation.copy()

            changed_scale, new_scale = imgui.drag_float3(
                "Scale", comp.local.scale.tolist(), 0.1)
            if changed_scale:
                comp.local.scale = vec3(*new_scale)

            imgui.tree_pop()

    elif isinstance(comp, DirectionalLight):
        if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
            changed_enabled, new_enabled = imgui.checkbox(
                "Enabled", comp.enabled)
            if changed_enabled:
                comp.enabled = new_enabled
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
            changed_near, new_near = imgui.drag_float("Near plane", comp.near, 0.01, 0.01, 1000.0)
            if changed_near:
                comp.near = new_near
            changed_far, new_far = imgui.drag_float("Far plane", comp.far, 1.0, 0.01, 1000.0)
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
                        ("Normal", "normal_map"),
                        ("Roughness", "roughness_map"),
                        ("Metallic", "metallic_map")
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

            changed_yscale, new_yscale = imgui.drag_float("Y scale", comp.y_scale, 0.001, 0.0001, 10.0, "%.4f")
            if changed_yscale:
                comp.y_scale = new_yscale
                comp.dirty = True

            imgui.text_disabled(f"Iterations: {comp.iterations}")

            if imgui.button("+ Add optimizer"):
                spawn_pos_local = vec3(0.0, 0.0, 0.0)

                optimizer_mesh = AssetSystem.create_immediate_mesh(assets_state, *generate_uv_sphere(radius=0.1, stacks=10, sectors=20))

                new_entity = registry.create_entity()
                registry.add_components(
                    new_entity,
                    EntityFlags(name=f"Optimizer {new_entity}"),
                    Transform(local=TransformData(position=spawn_pos_local)),
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

    elif isinstance(comp, SceneGeneratorState):
        if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
            _, comp.street_length = imgui.drag_float("Street length", comp.street_length, 1.0, 10.0, 1000.0)
            _, comp.building_count = imgui.drag_int("Buildings", comp.building_count, 1, 0, 100)
            _, comp.vehicle_count = imgui.drag_int("Vehicles", comp.vehicle_count, 1, 0, 50)

            if imgui.button("Regenerate scene"):
                comp.should_generate = True
            imgui.tree_pop()

    elif isinstance(comp, SceneAnimatorState):
        if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
            _, comp.animate_vehicles = imgui.checkbox("Animate vehicles", comp.animate_vehicles)
            imgui.tree_pop()

    else:
        disabled_bullet(comp_type.__name__)


def disabled_bullet(text):
    imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha * 0.5)
    disabled_color = imgui.get_style().color_(imgui.Col_.text_disabled)
    imgui.push_style_color(imgui.Col_.text, disabled_color)
    imgui.bullet_text(text)
    imgui.pop_style_color(1)
    imgui.pop_style_var(1)


def focus_camera_on_transform(
    camera_state: CameraState,
    camera: Camera,
    camera_transform: Transform,
    target_transform: Transform,
):
    camera_state.focal_point = target_transform.world.position
    camera_transform.local.position = camera_state.focal_point - camera_state.front * camera.focal_point_distance  # type: ignore
