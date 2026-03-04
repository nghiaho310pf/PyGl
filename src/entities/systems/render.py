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

    def attach_shader(self, shader: Shader):
        self.shader_globals.attach_to(shader)

    def update(self, window_size: tuple[int, int], time: float, delta_time: float):
        # == camera setup ==

        cameras = list(self.registry.view(Transform, Camera))
        if len(cameras) == 0:
            return

        width = math.ceil(window_size[0] / len(cameras))
        height = window_size[1]
        aspect_ratio = width / height if height > 0 else 1.0

        # == single light setup ==

        light_res = self.registry.get_singleton(Transform, PointLight)
        light_pos = np.array([0.0, 0.0, 0.0])
        light_color = np.array([1.0, 1.0, 1.0])

        if light_res:
            _, (light_transform, point_light) = light_res
            light_pos = light_transform.position
            light_color = point_light.color

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.01, 0.01, 0.01, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glCullFace(GL.GL_BACK)

        # == batch by shader ==

        shader_batches = {}
        for entity, (transform, visuals) in self.registry.view(Transform, Visuals):
            shader = visuals.material.shader
            mat = visuals.material

            if shader not in shader_batches:
                shader_batches[shader] = {}
            if mat not in shader_batches[shader]:
                shader_batches[shader][mat] = []

            shader_batches[shader][mat].append((transform, visuals))

        for camera_index in range(len(cameras)):
            (camera_id, (camera_transform, camera)) = cameras[camera_index]

            # == view/projection matrices ==

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

            # == draw ==

            GL.glViewport(camera_index * width, 0, width, height)

            for shader, material_group in shader_batches.items():
                shader.use()

                shader.set_vec3("u_LightPos", light_pos)
                shader.set_vec3("u_LightColor", light_color)

                for material, entities in material_group.items():
                    material.setup_properties()

                    for transform, visuals in entities:
                        model_matrix = math_utils.create_transformation_matrix(
                            transform.position, transform.rotation, transform.scale
                        )
                        shader.set_mat4("u_Model", model_matrix)
                        visuals.mesh.draw()