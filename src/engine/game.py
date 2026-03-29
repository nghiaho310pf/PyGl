from typing import Any, Type

from imgui_bundle import imgui, icons_fontawesome_6
import numpy as np

from engine.application import Application
from entities.components.camera import Camera
from entities.components.entity_flags import EntityFlags
from entities.components.point_light import PointLight
from entities.components.render_state import RenderState
from entities.components.transform import Transform
from entities.components.ui_state import UiState
from entities.components.visuals import Visuals
from entities.components.camera_state import CameraState
from entities.registry import Registry
from entities.systems.render import RenderSystem
from entities.systems.ui import UiSystem
from entities.systems.camera_control import CameraControlSystem
from entities.systems.icon_renderer import IconRendererSystem
from meshes.geometry.cube import generate_cube
from meshes.geometry.plane import generate_plane
from meshes.geometry.uv_sphere import generate_uv_sphere
from meshes.mesh import Mesh
from math_utils import vec3
from shading.material import Material, ShaderType


class Game(Application):
    def __init__(self, initial_window_width, initial_window_height):
        super().__init__(initial_window_width, initial_window_height)

        self.last_update = None

        self.registry = Registry()

        self.render_system = RenderSystem()
        self.ui_system = UiSystem()
        self.camera_control_system = CameraControlSystem()
        self.icon_renderer_system = IconRendererSystem()

        # == material setup ==
        mat_preview = Material(ShaderType.BlinnPhong, {
            "u_Albedo": [0.4, 0.9, 0.4],
            "u_Roughness": 0.5,
            "u_Reflectance": 0.1,
            "u_AO": 0.0,
        })
        mat_default = Material(ShaderType.BlinnPhong, {
            "u_Albedo": [0.3, 0.3, 0.3],
            "u_Roughness": 0.5,
            "u_Reflectance": 0.1,
            "u_AO": 0.0,
        })

        # == singleton entities required for above systems ==
        self.registry.add_components(
            self.registry.create_entity(),
            EntityFlags(is_internal=True),
            RenderState(),
            CameraState(),
        )
        self.registry.add_components(
            self.registry.create_entity(),
            EntityFlags(is_internal=True),
            UiState(default_material=mat_default),
            Transform(position=vec3(0.0, 0.0, 0.0)),
            Visuals(
                Mesh(np.array([], dtype=np.float32), np.array([], dtype=np.uint32)),
                material=mat_preview, enabled=False, is_internal=True
            ) 
        )

        # == demo setup ==

        mat_orange = Material(ShaderType.BlinnPhong, {
            "u_Albedo": [1.0, 0.318, 0.133],
            "u_Roughness": 0.6,
            "u_Reflectance": 0.25,
            "u_AO": 0.05,
        })
        mat_blue = Material(ShaderType.BlinnPhong, {
            "u_Albedo": [0.276, 0.481, 1.0],
            "u_Roughness": 0.6,
            "u_Reflectance": 0.25,
            "u_AO": 0.05,
        })
        mat_grey = Material(ShaderType.BlinnPhong, {
            "u_Albedo": [0.08, 0.08, 0.08],
            "u_Roughness": 0.6,
            "u_Reflectance": 0.01,
            "u_AO": 0.05,
        })

        uv_sphere_vertices, uv_sphere_indices = generate_uv_sphere(radius=0.5, stacks=20, sectors=40)
        cube_vertices, cube_indices = generate_cube(size=1.0)
        plane_vertices, plane_indices = generate_plane()

        # orange UV sphere entity
        e1 = self.registry.create_entity()
        self.registry.add_components(
            e1,
            EntityFlags(name="UV sphere"),
            Transform(position=vec3(0.0, 0.5, 0)),
            Visuals(Mesh(uv_sphere_vertices, uv_sphere_indices), mat_orange)
        )

        # blue cube entity
        e2 = self.registry.create_entity()
        self.registry.add_components(
            e2,
            EntityFlags(name="Cube"),
            Transform(position=vec3(-1.6, 0.5, 0), rotation=vec3(0.0, -57.0, 0.0)),
            Visuals(Mesh(cube_vertices, cube_indices), mat_blue)
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

        # == logic & graphics ==
        # a bit unorthodox to ECS because of calling special ImGui commands
        # here instead of in systems, but this is simpler.
        self.imgui_renderer.process_inputs()
        imgui.new_frame()
        self.camera_control_system.update(self.registry, now, dt)
        self.render_system.update(self.registry, self.get_window_size(), now, dt)
        self.icon_renderer_system.update(self.registry, self.get_window_size())
        self.ui_system.update(self.registry, now, dt)
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

        # == wrapup ==
        self.last_update = now
