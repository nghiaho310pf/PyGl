from typing import Any, Type

from imgui_bundle import imgui
import numpy as np

from engine.application import Application
from entities.components.camera import Camera
from entities.components.entity_name import EntityName
from entities.components.point_light import PointLight
from entities.components.rotated import Rotated
from entities.components.transform import Transform
from entities.components.visuals import Visuals
from entities.registry import Registry
from entities.systems.render import RenderSystem
from entities.systems.rotator import RotatorSystem
from geometry.geometry import generate_sphere, generate_cube_flat, generate_plane
from geometry.mesh import Mesh
from math_utils import vec3
from shading.material import Material
from shading.shaders import blinn_phong


class Game(Application):
    # == editor state ==
    selected_entity: int | None = None

    def __init__(self, initial_window_width, initial_window_height):
        super().__init__(initial_window_width, initial_window_height)

        self.registry = Registry()

        self.render_system = RenderSystem(self.registry)
        self.rotator_system = RotatorSystem(self.registry)

        # == demo setup ==

        shader = blinn_phong.make_shader()
        self.render_system.attach_shader(shader)

        mat_orange = Material(shader, {
            "u_Albedo": [1.0, 0.318, 0.133],
            "u_Roughness": 0.6,
            "u_Reflectance": 0.25,
            "u_AO": 0.05,
        })
        mat_blue = Material(shader, {
            "u_Albedo": [0.276, 0.481, 1.0],
            "u_Roughness": 0.6,
            "u_Reflectance": 0.25,
            "u_AO": 0.05,
        })
        mat_grey = Material(shader, {
            "u_Albedo": [0.08, 0.08, 0.08],
            "u_Roughness": 0.6,
            "u_Reflectance": 0.01,
            "u_AO": 0.05,
        })

        sphere_vertices, sphere_indices = generate_sphere(radius=0.5, stacks=20, sectors=40)
        cube_vertices, cube_indices = generate_cube_flat(size=1.0)
        plane_vertices, plane_indices = generate_plane()

        # orange sphere entity
        e1 = self.registry.create_entity()
        self.registry.add_components(
            e1,
            EntityName("Sphere"),
            Transform(position=vec3(0.0, 0.5, 0)),
            Visuals(Mesh(sphere_vertices, sphere_indices), mat_orange)
        )

        # blue cube entity
        e2 = self.registry.create_entity()
        self.registry.add_components(
            e2,
            EntityName("Cube"),
            Transform(position=vec3(0.0, 1.5, 0)),
            Visuals(Mesh(cube_vertices, cube_indices), mat_blue),
            Rotated(delta=vec3(0.0, 90.0, 0.0))
        )

        # floor entity
        e3 = self.registry.create_entity()
        self.registry.add_components(
            e3,
            EntityName("Plane"),
            Transform(position=vec3(0.0, 0.0, 0.0)),
            Visuals(Mesh(plane_vertices, plane_indices), mat_grey)
        )

        # camera entity
        c = self.registry.create_entity()
        self.registry.add_components(
            c,
            EntityName("Camera 1"),
            Transform(position=vec3(0.0, 2.4, 5.0), rotation=vec3(-22.0, -100.0, 0.0)),
            Camera()
        )
        c = self.registry.create_entity()
        self.registry.add_components(
            c,
            EntityName("Camera 2"),
            Transform(position=vec3(5.0, 2.4, 0.0), rotation=vec3(-22.0, -167.0, 0.0)),
            Camera()
        )

        # point light entities
        c1 = self.registry.create_entity()
        self.registry.add_components(
            c1,
            EntityName("Main light"),
            Transform(position=vec3(1.2, 4.0, 1.2)),
            PointLight(strength=np.float32(220.0))
        )

        c2 = self.registry.create_entity()
        self.registry.add_components(
            c2,
            EntityName("Sub light"),
            Transform(position=vec3(-1.0, 5.0, -1.0)),
            PointLight(strength=np.float32(60.0))
        )

        self.fps_tracker = 0.0

        self.last_update = None

    def render(self):
        # == preparation ==
        now = self.get_time()
        if self.last_update is None:
            self.last_update = now
        dt = now - self.last_update

        # == logic ==
        if dt > 0:
            self.fps_tracker = self.fps_tracker * 0.9 + (1.0 / dt) * 0.1

        self.imgui_renderer.process_inputs()

        self.rotator_system.update(now, dt)

        # == graphics ==
        imgui.new_frame()
        self.render_system.update(self.get_window_size(), now, dt)
        self.render_ui()
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

        # == wrapup ==
        self.last_update = now

    def render_ui(self):
        # == entity list window ==
        imgui.begin("Entities")
        
        content_max_x = imgui.get_cursor_pos().x + imgui.get_content_region_avail().x
        ICON_PRIORITY = {
            Camera: "\uf03d",
            PointLight: "\uf0eb",
            Visuals: "\uf1b2",
            Rotated: "\uf01e",
            Transform: "\uf047",
        }
        FIXED_ICON_WIDTH = 26 

        for entity_id, components in self.registry.view_all():
            entity_name: EntityName | None = components.get(EntityName)
            display_name = entity_name.name if (entity_name and entity_name.name) else "Entity"

            display_icon = None
            for comp_type, icon in ICON_PRIORITY.items():
                if comp_type in components:
                    display_icon = icon
                    break

            start_x = imgui.get_cursor_pos_x()
            
            if display_icon:
                imgui.text_disabled(display_icon)
                imgui.same_line(int(start_x) + FIXED_ICON_WIDTH)
            else:
                imgui.set_cursor_pos_x(int(start_x) + FIXED_ICON_WIDTH)

            id_str = f"#{entity_id}"
            selectable_label = f"{display_name}###entity_{entity_id}"

            clicked, _ = imgui.selectable(selectable_label, self.selected_entity == entity_id)
            if clicked:
                self.selected_entity = entity_id

            id_width = imgui.calc_text_size(id_str).x
            align_x = content_max_x - id_width

            imgui.same_line(int(align_x))
            imgui.text_disabled(id_str)

        imgui.end()

        # == inspector window ==
        if self.selected_entity is not None:
            selected_components = self.registry.get_components(self.selected_entity)

            entity_name = selected_components.get(EntityName)
            if entity_name is not None and entity_name.name is not None:
                entity_label = f"{entity_name.name} (#{self.selected_entity})"
            else:
                entity_label = f"Entity #{self.selected_entity}"

            # 3. Use '###' to give it a dynamic title but a static ID ("InspectorWindow")
            window_title = f"Inspector: {entity_label}###InspectorWindow"

            inspector_expanded, inspector_open = imgui.begin(window_title, True)

            if not inspector_open:
                self.selected_entity = None
            elif inspector_expanded:
                for comp_type, component in selected_components.items():
                    self.draw_component_properties(self.selected_entity, comp_type, component)

            imgui.end()

    def draw_component_properties(self, entity_id: int, comp_type: Type[Any], comp: Any):
        if isinstance(comp, EntityName):
            changed_name, new_name = imgui.input_text("Name", comp.name if comp.name is not None else "")
            if changed_name:
                comp.name = new_name if new_name != "" else None
        
        elif isinstance(comp, Transform):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_pos, new_pos = imgui.drag_float3("Position", comp.position, 0.1)
                if changed_pos:
                    comp.position = vec3(*new_pos)

                changed_rot, new_rot = imgui.drag_float3("Rotation", comp.rotation, 1.0)
                if changed_rot:
                    comp.rotation = vec3(*new_rot)

                changed_scale, new_scale = imgui.drag_float3("Scale", comp.scale, 0.1)
                if changed_scale:
                    comp.scale = vec3(*new_scale)
                imgui.tree_pop()

        elif isinstance(comp, PointLight):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_enabled, new_enabled = imgui.checkbox("Enabled", comp.enabled)
                if changed_enabled:
                    comp.enabled = new_enabled
                changed_color, new_color = imgui.color_edit3("Color", comp.color)
                if changed_color:
                    comp.color = np.array(new_color)
                changed_strength, new_strength = imgui.drag_float("Strength", float(comp.strength), 1, 0.0, 1000.0)
                if changed_strength:
                    comp.strength = np.float32(new_strength)
                imgui.tree_pop()
                
        elif isinstance(comp, Camera):
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                imgui.begin_disabled(self.render_system.target_camera == entity_id)
                changed_targeted, new_targeted = imgui.checkbox("Main camera", self.render_system.target_camera == entity_id)
                if changed_targeted:
                    self.render_system.target_camera = entity_id
                imgui.end_disabled()

                changed_fov, new_fov = imgui.drag_float("FOV", comp.fov, 1.0, 10.0, 150.0)
                if changed_fov:
                    comp.fov = new_fov
                    
                changed_near, new_near = imgui.drag_float("Near Plane", comp.near, 0.01)
                if changed_near:
                    comp.near = new_near
                    
                changed_far, new_far = imgui.drag_float("Far Plane", comp.far, 1.0)
                if changed_far:
                    comp.far = new_far
                imgui.tree_pop()
        
        elif isinstance(comp, Visuals) and not comp.is_internal:
            if imgui.tree_node_ex(comp_type.__name__, imgui.TreeNodeFlags_.default_open):
                changed_enabled, new_enabled = imgui.checkbox("Shown", comp.enabled)
                if changed_enabled:
                    comp.enabled = new_enabled

                imgui.tree_pop()

            # hardcode "Material", there might be a separate mesh-related dropdown later
            if imgui.tree_node_ex("Material", imgui.TreeNodeFlags_.default_open):
                # for now just a hardcoded whitelist of the parameters

                albedo = comp.material.properties["u_Albedo"]
                if albedo is not None:
                    changed_albedo, new_albedo = imgui.color_edit3("Albedo", albedo)
                    if changed_albedo:
                        comp.material.properties["u_Albedo"] = new_albedo
                
                roughness = comp.material.properties["u_Roughness"]
                if roughness is not None:
                    changed_roughness, new_roughness = imgui.slider_float("Roughness", roughness, 0.0, 1.0)
                    if changed_roughness:
                        comp.material.properties["u_Roughness"] = new_roughness

                reflectance = comp.material.properties["u_Reflectance"]
                if reflectance is not None:
                    changed_reflectance, new_reflectance = imgui.slider_float("Reflectance", reflectance, 0.0, 1.0)
                    if changed_reflectance:
                        comp.material.properties["u_Reflectance"] = new_reflectance

                ao = comp.material.properties["u_AO"]
                if ao is not None:
                    changed_ao, new_ao = imgui.slider_float("AO", ao, 0.0, 1.0)
                    if changed_ao:
                        comp.material.properties["u_AO"] = new_ao

                imgui.tree_pop()

        else:
            self.disabled_bullet(comp_type.__name__)

    def disabled_bullet(self, text):
        imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha * 0.5)
        disabled_color = imgui.get_style().color_(imgui.Col_.text_disabled)
        imgui.push_style_color(imgui.Col_.text, disabled_color)
        imgui.bullet_text(text)
        imgui.pop_style_color(1)
        imgui.pop_style_var(1)