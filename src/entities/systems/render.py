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
from shading.shaders import depth_shader, tf2_ggx_smith



class RenderSystem:
    def __init__(self, registry: Registry):
        self.registry = registry
        self.shader_globals = ShaderGlobals()

        self.depth_shader = depth_shader.make_shader()

        self.SHADOW_WIDTH = 1024
        self.SHADOW_HEIGHT = 1024

        for entity, (point_light,) in self.registry.view(PointLight):
            self._setup_shadow_map(point_light)

    def _setup_shadow_map(self, point_light: PointLight):
        # Create FBO
        point_light.shadow_map_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, point_light.shadow_map_fbo)

        # Create depth texture
        point_light.shadow_map_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, point_light.shadow_map_texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT, self.SHADOW_WIDTH, self.SHADOW_HEIGHT, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        # Use a border color for sampling outside the shadow map
        border_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        GL.glTexParameterfv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR, border_color)

        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, point_light.shadow_map_texture, 0)
        GL.glDrawBuffer(GL.GL_NONE) # No color buffer is needed
        GL.glReadBuffer(GL.GL_NONE)

        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            print("ERROR::FRAMEBUFFER:: Shadow FBO is not complete!")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

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
            point_light_positions.append(point_light_transform.position)
            point_light_colors.append(point_light.color)
        
        num_lights = len(point_light_positions)
        MAX_LIGHTS = 4 # Should match the shader's MAX_LIGHTS
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

        # == Shadow Pass ==
        GL.glViewport(0, 0, self.SHADOW_WIDTH, self.SHADOW_HEIGHT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glCullFace(GL.GL_FRONT) # Avoid Peter Panning

        light_space_matrices = []
        shadow_map_textures = []
        shadow_map_texture_units = []
        texture_unit_counter = 0

        for light_entity, (light_transform, point_light) in self.registry.view(Transform, PointLight):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, point_light.shadow_map_fbo)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

            light_projection = math_utils.create_perspective_projection(90.0, 1.0, 0.1, 100.0) # A 90-degree perspective projection for point light shadow
            light_view = math_utils.create_look_at(
                light_transform.position,
                np.array([0.0, 0.0, 0.0]), # Look at the center of the scene for now
                np.array([0.0, 1.0, 0.0])
            )
            point_light.light_projection_matrix = light_projection
            point_light.light_view_matrix = light_view

            light_space_matrix = np.dot(light_projection, light_view)
            light_space_matrices.append(light_space_matrix)
            shadow_map_textures.append(point_light.shadow_map_texture)
            shadow_map_texture_units.append(texture_unit_counter)

            self.depth_shader.use()
            self.depth_shader.set_mat4("u_LightSpaceMatrix", light_space_matrix)

            for entity, (transform, visuals) in self.registry.view(Transform, Visuals):
                model_matrix = math_utils.create_transformation_matrix(
                    transform.position, transform.rotation, transform.scale
                )
                self.depth_shader.set_mat4("u_Model", model_matrix)
                visuals.mesh.draw()

            texture_unit_counter += 1

        GL.glCullFace(GL.GL_BACK) # Reset culling
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0) # Unbind FBO
        GL.glViewport(0, 0, width, height) # Reset viewport

        self.shader_globals.update(proj_matrix, view_matrix, camera_transform.position, time)

        # == drawing entities ==

        GL.glClearColor(0.01, 0.01, 0.01, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        for entity, (transform, visuals) in self.registry.view(Transform, Visuals):
            visuals.material.use()

            shader = visuals.material.shader
            shader.set_vec3_array("u_LightPos", point_light_positions)
            shader.set_vec3_array("u_LightColor", point_light_colors)
            shader.set_int("u_NumLights", num_lights)


            model_matrix = math_utils.create_transformation_matrix(
                transform.position, transform.rotation, transform.scale
            )

            shader.set_mat4("u_Model", model_matrix)

            visuals.mesh.draw()
