import math

import numpy as np
from OpenGL import GL

import math_utils
from entities.components.camera import Camera
from entities.components.point_light import PointLight
from entities.components.transform import Transform
from entities.components.visuals import Visuals
from entities.registry import Registry
from shading.shader import Shader, ShaderGlobals
from shading.shaders import depth_shader


class RenderSystem:
    def __init__(self, registry: Registry):
        self.registry = registry
        self.shader_globals = ShaderGlobals()

        self.depth_shader = depth_shader.make_shader()

    def attach_shader(self, shader: Shader):
        self.shader_globals.attach_to(shader)

    def update(self, window_size: tuple[int, int], time: float, delta_time: float):
        # == camera setup ==

        width, height = window_size
        aspect_ratio = width / height if height > 0 else 1.0

        r = self.registry.get_singleton(Transform, Camera)
        if r is None:
            return
        camera_entity, (camera_transform, camera) = r

        point_light_positions = []
        point_light_colors = []

        for light_entity, (point_light_transform, point_light) in self.registry.view(Transform, PointLight):
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

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, width, height)

        GL.glClearColor(0.004, 0.004, 0.004, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.shader_globals.update(proj_matrix, view_matrix, camera_transform.position, time)

        shader_batches = {}
        for entity, (transform, visuals) in self.registry.view(Transform, Visuals):
            if not visuals.enabled:
                continue

            shader = visuals.material.shader
            mat = visuals.material

            if shader not in shader_batches:
                shader_batches[shader] = {}
            if mat not in shader_batches[shader]:
                shader_batches[shader][mat] = []

            shader_batches[shader][mat].append((transform, visuals))

        for shader, material_group in shader_batches.items():
            shader.use()

            shader.set_vec3_array("u_LightPos", point_light_positions)
            shader.set_vec3_array("u_LightColor", point_light_colors)
            shader.set_int("u_NumLights", num_lights)

            for material, entities in material_group.items():
                material.setup_properties()

                for transform, visuals in entities:
                    model_matrix = math_utils.create_transformation_matrix(
                        transform.position, transform.rotation, transform.scale
                    )
                    shader.set_mat4("u_Model", model_matrix)

                    visuals.mesh.draw()
