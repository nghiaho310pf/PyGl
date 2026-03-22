from imgui_bundle import imgui

from engine.application import Application
from entities.components.camera import Camera
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
from shading.shaders import tf2_ggx_smith


class Game(Application):
    def __init__(self, initial_window_width, initial_window_height):
        super().__init__(initial_window_width, initial_window_height)

        self.registry = Registry()

        self.render_system = RenderSystem(self.registry)
        self.rotator_system = RotatorSystem(self.registry)

        # == demo setup ==

        shader = tf2_ggx_smith.make_shader()
        self.render_system.attach_shader(shader)

        mat_orange = Material(shader, {
            "u_Albedo": [1.0, 0.318, 0.133],
            "u_Metallic": 0.3,
            "u_Roughness": 0.6,
            "u_Reflectance": 0.25,
            "u_Translucency": 0.0,
            "u_AO": 0.05,
        })
        mat_blue = Material(shader, {
            "u_Albedo": [0.276, 0.481, 1.0],
            "u_Metallic": 0.7,
            "u_Roughness": 0.6,
            "u_Reflectance": 0.25,
            "u_Translucency": 0.0,
            "u_AO": 0.05,
        })
        mat_grey = Material(shader, {
            "u_Albedo": [0.08, 0.08, 0.08],
            "u_Metallic": 0.7,
            "u_Roughness": 0.6,
            "u_Reflectance": 0.0,
            "u_Translucency": 0.0,
            "u_AO": 0.05,
        })

        sphere_vertices, sphere_indices = generate_sphere(radius=0.5, stacks=20, sectors=40)
        cube_vertices, cube_indices = generate_cube_flat(size=1.0)
        plane_vertices, plane_indices = generate_plane()

        # orange sphere entity
        e1 = self.registry.create_entity()
        self.registry.add_components(
            e1,
            Transform(position=vec3(0.0, 0.5, 0)),
            Visuals(Mesh(sphere_vertices, sphere_indices), mat_orange)
        )

        # blue cube entity
        e2 = self.registry.create_entity()
        self.registry.add_components(
            e2,
            Transform(position=vec3(0.0, 1.5, 0)),
            Visuals(Mesh(cube_vertices, cube_indices), mat_blue),
            Rotated(delta=vec3(0.0, 90.0, 0.0))
        )

        # floor entity
        e3 = self.registry.create_entity()
        self.registry.add_components(
            e3,
            Transform(position=vec3(0.0, 0.0, 0.0)),
            Visuals(Mesh(plane_vertices, plane_indices), mat_grey)
        )

        # camera entity
        c = self.registry.create_entity()
        self.registry.add_components(
            c,
            Transform(position=vec3(0.0, 2.4, 5.0), rotation=vec3(-22.0, -100.0, 0.0)),
            Camera()
        )

        # point light entities
        c = self.registry.create_entity()
        self.registry.add_components(
            c,
            Transform(position=vec3(1.2, 4.0, 1.2)),
            PointLight(color=vec3(220.0, 220.0, 220.0))
        )

        self.tweakable_light = c

        c = self.registry.create_entity()
        self.registry.add_components(
            c,
            Transform(position=vec3(-1.0, 5.0, -1.0)),
            PointLight(color=vec3(60.0, 60.0, 60.0), radius=0.1)
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
        imgui.begin("Engine Debug")
        
        imgui.text(f"FPS: {self.fps_tracker:.1f}")
        imgui.separator()

        r = self.registry.get_components(self.tweakable_light, PointLight)
        if r is None:
            return
        (light_component, ) = r

        current_color = (
            light_component.color[0] / 255.0,
            light_component.color[1] / 255.0,
            light_component.color[2] / 255.0
        )
        
        changed, new_color = imgui.color_edit3("Light Color", current_color)
        
        if changed:
            light_component.color = vec3(
                new_color[0] * 255.0,
                new_color[1] * 255.0,
                new_color[2] * 255.0
            )
            
        changed_radius, new_radius = imgui.slider_float("Light Radius", light_component.radius, 0.0, 1.0)
        if changed_radius:
            light_component.radius = new_radius

        imgui.end()
