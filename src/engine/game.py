from typing import Any, Type

from imgui_bundle import imgui
import numpy as np

from engine.application import Application
from entities.components.camera import Camera
from entities.components.disposal import Disposal
from entities.components.entity_flags import EntityFlags
from entities.components.spawner_state import SpawnerState
from entities.components.visuals.assets import AssetsState
from entities.components.ui.icon_render_state import IconRenderState
from entities.components.ui.gizmo_state import GizmoState
from entities.components.directional_light import DirectionalLight
from entities.components.render_state import RenderState
from entities.components.transform import Transform
from entities.components.ui.ui_state import UiState
from entities.components.visuals.visuals import Visuals
from entities.components.camera_state import CameraState
from entities.components.street_scene.scene_animator_state import SceneAnimatorState
from entities.components.street_scene.scene_generator_state import SceneGeneratorState
from entities.registry import Registry
from entities.systems.disposal import DisposalSystem
from entities.systems.function_surface import FunctionSurfaceSystem
from entities.systems.gradient_descent import GradientDescentSurfaceSystem
from entities.systems.render import RenderSystem
from entities.systems.assets import AssetSystem
from entities.systems.spawner import SpawnerSystem
from entities.systems.ui.system import UiSystem
from entities.systems.camera import CameraSystem
from entities.systems.icon_render import IconRenderSystem
from entities.systems.bounding_box_render import BoundingBoxRenderSystem
from entities.systems.gizmo import GizmoSystem
from entities.systems.transform_inheritance import TransformInheritanceSystem
from entities.systems.street_scene.scene_generator import SceneGeneratorSystem
from entities.systems.street_scene.scene_animator import SceneAnimatorSystem
from entities.components.visuals.assets import AssetsState
from math_utils import float1, quaternion_from_euler, vec3
from entities.components.visuals.material import Material


class Game(Application):
    def __init__(self, initial_window_width, initial_window_height):
        super().__init__(initial_window_width, initial_window_height)

        self.last_update = None

        self.registry = Registry()

        # unorthodox system with OpenGL state that we need to make an instance of
        self.render_system = RenderSystem()

        # == singleton components & entities required for above systems ==
        assets_state = AssetsState()
        spawner_state = SpawnerState()

        # == material setup ==
        mat_preview = Material(
            albedo=vec3(0.4, 0.9, 0.4),
            roughness=float1(0.5),
            metallic=float1(0.2),
            reflectance=float1(0.1),
            ao=float1(0.1),
        )
        mat_default = Material(
            albedo=vec3(0.3, 0.3, 0.3),
            metallic=float1(0.2),
            roughness=float1(0.5),
            reflectance=float1(0.1),
            ao=float1(0.1),
        )

        # preview entity for adding meshes
        preview_entity = self.registry.create_entity()
        self.registry.add_components(
            preview_entity,
            EntityFlags(is_internal=True),
            Transform(position=vec3(0.0, 0.0, 0.0)),
            Visuals(
                AssetSystem.create_immediate_mesh(assets_state, np.array([], dtype=np.float32), np.array([], dtype=np.uint32)),
                material=mat_preview, enabled=False, is_internal=True
            )
        )
        # cheap trick: this is a singleton as a child of the selected entity,
        # so that DisposalSystem can do the work of getting rid of dangling IDs for us.
        # this would instead be a tagging component if we had to implement selecting
        # multiple entities, but we just need 1.
        selection_child_entity = self.registry.create_entity()
        self.registry.add_components(
            selection_child_entity,
            EntityFlags(is_internal=True, dispose_alongside_parent=False),
        )
        # similar trick to the above for camera & camera control state
        camera_state_entity = self.registry.create_entity()
        self.registry.add_components(
            camera_state_entity,
            EntityFlags(is_internal=True, dispose_alongside_parent=False),
            CameraState(),
        )
        # admin entity for overarching singletons
        self.registry.add_components(
            self.registry.create_entity(),
            EntityFlags(is_internal=True),

            Disposal(),

            UiState(
                preview_entity=preview_entity,
                selection_child_entity=selection_child_entity,
                default_material=mat_default
            ),
            assets_state,
            spawner_state,
            RenderState(),
            IconRenderState(),
            GizmoState(),
        )

        # camera entity
        c = self.registry.create_entity()
        self.registry.add_components(
            c,
            EntityFlags(name="Camera 1"),
            Transform(position=vec3(0.0, 2.4, 5.0), rotation=quaternion_from_euler(vec3(-22.0, 0.0, 0.0))),
            Camera(far=200)
        )

        # light entity
        l1 = self.registry.create_entity()
        self.registry.add_components(
            l1,
            EntityFlags(name="Light"),
            DirectionalLight(
                rotation=vec3(170.0, -110.0, 130.0),
                strength=float1(20.0)
            )
        )

        # street scene
        s = self.registry.create_entity()
        self.registry.add_components(
            s,
            EntityFlags(name="Street scene", is_internal=False),
            Transform(),
            SceneGeneratorState(should_generate=True),
            SceneAnimatorState()
        )

    def render(self):
        # == preparation ==
        now = self.get_time()
        if self.last_update is None:
            self.last_update = now
        dt = now - self.last_update

        window_size = self.get_window_size()

        # == logic & graphics ==
        # a bit unorthodox to ECS because of calling special ImGui commands
        # here instead of in systems, but this is simpler.
        self.imgui_renderer.process_inputs()
        imgui.new_frame()

        GizmoSystem.update(self.registry, window_size)
        SpawnerSystem.update(self.registry)
        SceneGeneratorSystem.update(self.registry)
        SceneAnimatorSystem.update(self.registry, dt)
        GradientDescentSurfaceSystem.update(self.registry, now, dt)
        FunctionSurfaceSystem.update(self.registry, now, dt)
        TransformInheritanceSystem.update(self.registry)
        CameraSystem.update(self.registry, window_size, now, dt)
        UiSystem.update(self.registry, now, dt)
        AssetSystem.update(self.registry)
        self.render_system.update(self.registry, window_size, now, dt)
        BoundingBoxRenderSystem.update(self.registry)
        IconRenderSystem.update(self.registry, window_size)

        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

        # == disposal ==
        DisposalSystem.update(self.registry)

        # == wrapup ==
        self.last_update = now
