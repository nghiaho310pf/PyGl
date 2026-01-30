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


class RenderSystem:
    def __init__(self, registry: Registry):
        self.registry = registry
        self.shader_globals = ShaderGlobals()

        # TODO: should this really be here?
        GL.glFrontFace(GL.GL_CCW)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)

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

        r = self.registry.get_singleton(Transform, PointLight)
        if r is None:
            return
        light_entity, (point_light_transform, point_light) = r

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

        self.shader_globals.update(proj_matrix, view_matrix, camera_transform.position, time)

        # == drawing entities ==

        GL.glClearColor(0.01, 0.01, 0.01, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        for entity, (transform, visuals) in self.registry.view(Transform, Visuals):
            visuals.material.use()

            shader = visuals.material.shader
            shader.set_vec3("u_LightPos", point_light_transform.position)
            shader.set_vec3("u_LightColor", point_light.color)

            model_matrix = math_utils.create_transformation_matrix(
                transform.position, transform.rotation, transform.scale
            )

            shader.set_mat4("u_Model", model_matrix)

            visuals.mesh.draw()
