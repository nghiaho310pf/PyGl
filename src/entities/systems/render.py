import math

import numpy as np
from OpenGL import GL
from numba import njit, float32, void

from entities.components.camera import Camera
from entities.components.point_light import PointLight
from entities.components.transform import Transform
from entities.components.visuals import Visuals
from entities.registry import Registry
from math_utils import normalize, compute_look_at_matrix, compute_projection_matrix, compute_transformation_matrix, vec3
from shading.shader import Shader, ShaderGlobals
from shading.shaders import depth_shader


@njit(void(float32[::1], float32[:, ::1], float32[:, ::1], float32, float32[::1], float32[::1], float32, float32,
           float32), cache=True, parallel=False)
def _update_matrices(world_up, view_matrix, projection_matrix, aspect_ratio, camera_position, camera_rotation,
                     camera_fov, camera_near, camera_far):
    # Use math.radians for scalars to avoid array overhead
    pitch_rad = math.radians(camera_rotation[0])
    yaw_rad = math.radians(camera_rotation[1])
    roll_rad = math.radians(camera_rotation[2])

    # Explicitly float32 to match your 'normalize' signature
    front_vec = np.array([
        math.cos(yaw_rad) * math.cos(pitch_rad),
        math.sin(pitch_rad),
        math.sin(yaw_rad) * math.cos(pitch_rad)
    ], dtype=np.float32)

    front = normalize(front_vec)

    # np.cross usually returns the same dtype as input
    right = normalize(np.cross(front, world_up))

    # Calculate Up
    up_raw = np.cross(right, front)
    up = normalize(up_raw) * np.float32(math.cos(roll_rad)) + right * np.float32(math.sin(roll_rad))

    target = camera_position + front

    compute_look_at_matrix(camera_position, target, up, view_matrix)

    # Ensure scalars are cast if necessary, though Numba usually handles scalar casting fine
    compute_projection_matrix(
        np.float32(camera_fov),
        np.float32(aspect_ratio),
        np.float32(camera_near),
        np.float32(camera_far),
        projection_matrix
    )


class RenderSystem:
    def __init__(self, registry: Registry):
        self.registry = registry
        self.shader_globals = ShaderGlobals()

        self.depth_shader = depth_shader.make_shader()

        self.SHADOW_WIDTH = 768
        self.SHADOW_HEIGHT = 768

        # numpy fields for hopefully reducing garbage generation
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.view_matrix = np.identity(4, dtype=np.float32)
        self.projection_matrix = np.zeros((4, 4), dtype=np.float32)
        self.model_matrix = np.identity(4, dtype=np.float32)

        self.light_projection = np.zeros((4, 4), dtype=np.float32)
        self.light_view = np.identity(4, dtype=np.float32)

    def _setup_shadow_map(self, point_light: PointLight):
        # Create FBO
        point_light.shadow_map_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, point_light.shadow_map_fbo)

        # Create depth cube map texture
        point_light.shadow_map_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, point_light.shadow_map_texture)
        for i in range(6):
            GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL.GL_DEPTH_COMPONENT,
                            self.SHADOW_WIDTH, self.SHADOW_HEIGHT, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)

        GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, point_light.shadow_map_texture, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def attach_shader(self, shader: Shader):
        self.shader_globals.attach_to(shader)

    def update(self, window_size: tuple[int, int], time: float, delta_time: float):
        # == shadow map setup ==

        for entity, (point_light,) in self.registry.view(PointLight):
            if point_light.shadow_map_fbo == 0:
                self._setup_shadow_map(point_light)

        # == camera setup ==

        width, height = window_size
        aspect_ratio = np.float32(width / height) if height > 0 else np.float32(1.0)

        r = self.registry.get_singleton(Transform, Camera)
        if r is None:
            return

        camera_entity, (camera_transform, camera) = r

        point_light_positions = []
        point_light_colors = []

        for light_entity, (point_light_transform, point_light) in self.registry.view(Transform, PointLight):
            point_light_positions.append(point_light_transform.position)
            point_light_colors.append(point_light.color)

        num_lights = len(point_light_positions)
        MAX_LIGHTS = 4  # Should match the shader's MAX_LIGHTS
        if num_lights > MAX_LIGHTS:
            point_light_positions = point_light_positions[:MAX_LIGHTS]
            point_light_colors = point_light_colors[:MAX_LIGHTS]
            num_lights = MAX_LIGHTS

        _update_matrices(
            self.world_up, self.view_matrix, self.projection_matrix,
            aspect_ratio,
            camera_transform.position, camera_transform.rotation,
            camera.fov, camera.near, camera.far
        )

        self.shader_globals.update(self.projection_matrix, self.view_matrix, camera_transform.position, time)

        # == shadow pass ==

        GL.glViewport(0, 0, self.SHADOW_WIDTH, self.SHADOW_HEIGHT)
        GL.glCullFace(GL.GL_FRONT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        compute_projection_matrix(90.0, 1.0, 0.1, 100.0, self.light_projection)
        shadow_map_textures = []
        shadow_map_texture_units = []
        texture_unit_counter = 0

        light_positions = []
        light_radii = []
        light_far_planes = []

        for light_entity, (light_transform, point_light) in self.registry.view(Transform, PointLight):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, point_light.shadow_map_fbo)

            shadow_dirs = [
                (vec3(1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0)),
                (vec3(-1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0)),
                (vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)),
                (vec3(0.0, -1.0, 0.0), vec3(0.0, 0.0, -1.0)),
                (vec3(0.0, 0.0, 1.0), vec3(0.0, -1.0, 0.0)),
                (vec3(0.0, 0.0, -1.0), vec3(0.0, -1.0, 0.0)),
            ]

            point_light.light_view_matrices = []

            self.depth_shader.use()
            self.depth_shader.set_float("u_FarPlane", 100.0)
            self.depth_shader.set_vec3("u_LightPos", light_transform.position)

            for i, (target_dir, up_dir) in enumerate(shadow_dirs):
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                          GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                          point_light.shadow_map_texture, 0)
                GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

                compute_look_at_matrix(light_transform.position, light_transform.position + target_dir, up_dir,
                                       self.light_view)
                point_light.light_view_matrices.append(self.light_view)
                self.depth_shader.set_mat4("u_Projection", self.light_projection)
                self.depth_shader.set_mat4("u_View", self.light_view)

                for entity, (transform, visuals) in self.registry.view(Transform, Visuals):
                    compute_transformation_matrix(
                        transform.position, transform.rotation, transform.scale,
                        self.model_matrix
                    )
                    self.depth_shader.set_mat4("u_Model", self.model_matrix)
                    visuals.mesh.draw()

            light_positions.append(light_transform.position)
            light_radii.append(point_light.radius)
            light_far_planes.append(100.0)
            shadow_map_textures.append(point_light.shadow_map_texture)
            shadow_map_texture_units.append(texture_unit_counter)
            texture_unit_counter += 1

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, width, height)

        self.shader_globals.update(self.projection_matrix, self.view_matrix, camera_transform.position, time)

        # == drawing entities ==

        GL.glClearColor(0.01, 0.01, 0.01, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glCullFace(GL.GL_BACK)

        shader_batches = {}
        for entity, (transform, visuals) in self.registry.view(Transform, Visuals):
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
            shader.set_float_array("u_FarPlane", light_far_planes)
            shader.set_float_array("u_LightRadius", light_radii)

            for i, tex_unit in enumerate(shadow_map_texture_units):
                GL.glActiveTexture(GL.GL_TEXTURE0 + tex_unit)
                GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, shadow_map_textures[i])
            shader.set_int_array("u_ShadowMap", shadow_map_texture_units)

            for material, entities in material_group.items():
                material.setup_properties()

                for transform, visuals in entities:
                    compute_transformation_matrix(
                        transform.position, transform.rotation, transform.scale,
                        out=self.model_matrix
                    )
                    shader.set_mat4("u_Model", self.model_matrix)

                    visuals.mesh.draw()

        GL.glActiveTexture(GL.GL_TEXTURE0)
