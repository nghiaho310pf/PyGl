from typing import Any, Type

from imgui_bundle import imgui, icons_fontawesome_6
import numpy as np

from engine.application import Application
from entities.components.camera import Camera
from entities.components.entity_flags import EntityFlags
from entities.components.point_light import PointLight
from entities.components.render_state import RenderState
from entities.components.rotated import Rotated
from entities.components.transform import Transform
from entities.components.ui_state import UiState
from entities.components.visuals import Visuals
from entities.registry import Registry
from entities.systems.render import RenderSystem
from entities.systems.rotator import RotatorSystem
from entities.systems.ui import UiSystem
from geometry.geometry import generate_sphere, generate_cube_flat, generate_plane
from geometry.mesh import Mesh
from math_utils import vec3
from shading.material import Material
from shading.shaders import blinn_phong


class Game(Application):
    def __init__(self, initial_window_width, initial_window_height):
        super().__init__(initial_window_width, initial_window_height)

        self.last_update = None

        self.registry = Registry()

        self.render_system = RenderSystem()
        self.ui_system = UiSystem()
        self.rotator_system = RotatorSystem()

        # == singleton entities required for above systems ==
        self.registry.add_components(
            self.registry.create_entity(),
            EntityFlags(is_internal=True),
            RenderState()
        )
        self.registry.add_components(
            self.registry.create_entity(),
            EntityFlags(is_internal=True),
            UiState()
        )

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
            EntityFlags(name="Sphere"),
            Transform(position=vec3(0.0, 0.5, 0)),
            Visuals(Mesh(sphere_vertices, sphere_indices), mat_orange)
        )

        # blue cube entity
        e2 = self.registry.create_entity()
        self.registry.add_components(
            e2,
            EntityFlags(name="Cube"),
            Transform(position=vec3(0.0, 1.5, 0)),
            Visuals(Mesh(cube_vertices, cube_indices), mat_blue),
            Rotated(delta=vec3(0.0, 90.0, 0.0))
        )

        # floor entity
        e3 = self.registry.create_entity()
        self.registry.add_components(
            e3,
            EntityFlags(name="Plane"),
            Transform(position=vec3(0.0, 0.0, 0.0)),
            Visuals(Mesh(plane_vertices, plane_indices), mat_grey)
        )

        # camera entity
        c = self.registry.create_entity()
        self.registry.add_components(
            c,
            EntityFlags(name="Camera 1"),
            Transform(position=vec3(0.0, 2.4, 5.0), rotation=vec3(-22.0, -100.0, 0.0)),
            Camera()
        )
        c = self.registry.create_entity()
        self.registry.add_components(
            c,
            EntityFlags(name="Camera 2"),
            Transform(position=vec3(5.0, 2.4, 0.0), rotation=vec3(-22.0, -167.0, 0.0)),
            Camera()
        )

        # point light entities
        c1 = self.registry.create_entity()
        self.registry.add_components(
            c1,
            EntityFlags(name="Main light"),
            Transform(position=vec3(1.2, 4.0, 1.2)),
            PointLight(strength=np.float32(220.0))
        )

        c2 = self.registry.create_entity()
        self.registry.add_components(
            c2,
            EntityFlags(name="Sub light"),
            Transform(position=vec3(-1.0, 5.0, -1.0)),
            PointLight(strength=np.float32(60.0))
        )

    def render(self):
        # == preparation ==
        now = self.get_time()
        if self.last_update is None:
            self.last_update = now
        dt = now - self.last_update

        # == logic ==
        self.imgui_renderer.process_inputs()
        self.rotator_system.update(self.registry, now, dt)

        # == graphics ==
        self.render_system.update(self.registry, self.get_window_size(), now, dt)

        # == ui ==
        # a bit unorthodox to ECS because of calling special ImGui commands
        # here instead of in UiSystem. might or might not fix later
        imgui.new_frame()
        self.ui_system.update(self.registry, now, dt)
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

        # == wrapup ==
        self.last_update = now
