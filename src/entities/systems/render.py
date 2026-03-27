import math
from typing import Tuple

import numpy as np
from OpenGL import GL

import math_utils
from entities.components.render_state import RenderState, DrawMode
from entities.components.camera import Camera
from entities.components.point_light import PointLight
from entities.components.transform import Transform
from entities.components.visuals import Visuals
from entities.registry import Registry
from shading.material import Material, ShaderType
from shading.shader import Shader, ShaderGlobals
from shading.shaders import blinn_phong, gouraud, depth_shader, flat_shader


class RenderSystem:
    def __init__(self):
        # == unorthodox: global state ==
        self.shader_globals = ShaderGlobals()

        self.default_camera_transform = Transform()
        self.default_camera_component = Camera()

        self.flat_shader = flat_shader.make_shader()
        self.blinn_phong_shader = blinn_phong.make_shader()
        self.gouraud_shader = gouraud.make_shader()
        self.depth_shader = depth_shader.make_shader()
        self.attach_shader(self.flat_shader)
        self.attach_shader(self.blinn_phong_shader)
        self.attach_shader(self.gouraud_shader)
        self.attach_shader(self.depth_shader)

    def attach_shader(self, shader: Shader):
        self.shader_globals.attach_to(shader)

    def update(self, registry: Registry, window_size: tuple[int, int], time: float, delta_time: float):
        # == camera setup ==

        width, height = window_size
        aspect_ratio = width / height if height > 0 else 1.0

        camera_transform = self.default_camera_transform
        camera = self.default_camera_component

        r = registry.get_singleton(RenderState)
        if r is None:
            return
        render_state_entity, (render_state, ) = r

        if render_state.target_camera is not None:
            r = registry.get_components(render_state.target_camera, Transform, Camera)
            if r is None:
                render_state.target_camera = None
            else:
                camera_transform, camera = r

        if render_state.target_camera is None:
            r = registry.get_singleton(Transform, Camera)
            if r is not None:
                camera_entity, (camera_transform, camera) = r
                render_state.target_camera = camera_entity

        point_light_positions = []
        point_light_colors = []

        for light_entity, (point_light_transform, point_light) in registry.view(Transform, PointLight):
            if point_light.enabled:
                point_light_positions.append(point_light_transform.position)
                point_light_colors.append(point_light.color * point_light.strength)

        num_lights = len(point_light_positions)
        MAX_LIGHTS = 4  # Should match the shader's MAX_LIGHTS
        if num_lights > MAX_LIGHTS:
            point_light_positions = point_light_positions[:MAX_LIGHTS]
            point_light_colors = point_light_colors[:MAX_LIGHTS]
            num_lights = MAX_LIGHTS

        pitch_rad, yaw_rad, roll_rad = np.radians(camera_transform.rotation)
        front = math_utils.normalize(np.array([
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad)
        ]))
        right = math_utils.normalize(np.cross(front, np.array([0.0, 1.0, 0.0])))
        up = math_utils.normalize(np.cross(right, front)) * math.cos(roll_rad) + right * math.sin(roll_rad)

        target = camera_transform.position + front
        view_matrix = math_utils.create_look_at(camera_transform.position, target, up)

        proj_matrix = math_utils.create_perspective_projection(
            camera.fov, aspect_ratio, camera.near, camera.far
        )

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)

        if render_state.draw_mode == DrawMode.Wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        else:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, width, height)

        GL.glClearColor(0.004, 0.004, 0.004, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.shader_globals.update(proj_matrix, view_matrix, camera_transform.position, time)

        shader_batches: dict[ShaderType, dict[Material,
                                          list[Tuple[Transform, Visuals]]]] = {}
        for entity, (transform, visuals) in registry.view(Transform, Visuals):
            if not visuals.enabled:
                continue

            shader_type = visuals.material.shader_type
            mat = visuals.material

            if shader_type not in shader_batches:
                shader_batches[shader_type] = {}
            if mat not in shader_batches[shader_type]:
                shader_batches[shader_type][mat] = []

            shader_batches[shader_type][mat].append((transform, visuals))

        for shader_type, material_group in shader_batches.items():
            if render_state.draw_mode == DrawMode.DepthOnly:
                shader = self.depth_shader
            elif shader_type == ShaderType.Flat:
                shader = self.flat_shader
            elif shader_type == ShaderType.BlinnPhong:
                shader = self.blinn_phong_shader
            elif shader_type == ShaderType.Gouraud:
                shader = self.gouraud_shader
            else:
                shader = self.flat_shader

            shader.use()

            if render_state.draw_mode == DrawMode.DepthOnly:
                shader.set_float("u_Near", camera.near)
                shader.set_float("u_Far", camera.far)
            else:
                shader.set_vec3_array("u_LightPos", point_light_positions)
                shader.set_vec3_array("u_LightColor", point_light_colors)
                shader.set_int("u_NumLights", num_lights)

            for material, entities in material_group.items():
                if render_state.draw_mode != DrawMode.DepthOnly:
                    self.setup_shader_properties(shader, material)

                for transform, visuals in entities:
                    model_matrix = math_utils.create_transformation_matrix(
                        transform.position, transform.rotation, transform.scale
                    )
                    shader.set_mat4("u_Model", model_matrix)

                    visuals.mesh.draw()

    def setup_shader_properties(self, shader: Shader, material: Material):
        for name, value in material.properties.items():
            if isinstance(value, float):
                shader.set_float(name, value)
            elif isinstance(value, int):
                shader.set_int(name, value)
            elif isinstance(value, (list, tuple)):
                if len(value) == 3:
                    shader.set_vec3(name, value)
                elif len(value) == 4:
                    shader.set_vec4(name, value)