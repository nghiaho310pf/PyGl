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

        # == shadow pass ==

        GL.glViewport(0, 0, self.SHADOW_WIDTH, self.SHADOW_HEIGHT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        light_projection = math_utils.create_perspective_projection(90.0, 1.0, 0.1, 100.0)
        shadow_map_textures = []
        shadow_map_texture_units = []
        texture_unit_counter = 0

        light_positions = []
        light_radii = []
        light_far_planes = []

        for light_entity, (light_transform, point_light) in self.registry.view(Transform, PointLight):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, point_light.shadow_map_fbo)

            shadow_dirs = [
                (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
                (np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
                (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
                (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
                (np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])),
                (np.array([0.0, 0.0, -1.0]), np.array([0.0, -1.0, 0.0])),
            ]

            point_light.light_view_matrices = []
            point_light.light_projection_matrix = light_projection

            self.depth_shader.use()
            self.depth_shader.set_float("u_FarPlane", 100.0)
            self.depth_shader.set_vec3("u_LightPos", light_transform.position)

            for i, (target_dir, up_dir) in enumerate(shadow_dirs):
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                          GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                          point_light.shadow_map_texture, 0)
                GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

                light_view = math_utils.create_look_at(light_transform.position, light_transform.position + target_dir, up_dir)
                point_light.light_view_matrices.append(light_view)
                self.depth_shader.set_mat4("u_Projection", light_projection)
                self.depth_shader.set_mat4("u_View", light_view)

                for entity, (transform, visuals) in self.registry.view(Transform, Visuals):
                    model_matrix = math_utils.create_transformation_matrix(
                        transform.position, transform.rotation, transform.scale
                    )
                    self.depth_shader.set_mat4("u_Model", model_matrix)
                    visuals.mesh.draw()

            light_positions.append(light_transform.position)
            light_radii.append(point_light.radius)
            light_far_planes.append(100.0)
            shadow_map_textures.append(point_light.shadow_map_texture)
            shadow_map_texture_units.append(texture_unit_counter)
            texture_unit_counter += 1

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, width, height)

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

            for i, tex_unit in enumerate(shadow_map_texture_units):
                GL.glActiveTexture(GL.GL_TEXTURE0 + tex_unit)
                GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, shadow_map_textures[i])

            shader.set_int_array("u_ShadowMap", shadow_map_texture_units)
            shader.set_float_array("u_FarPlane", light_far_planes)
            shader.set_float_array("u_LightRadius", light_radii)

            model_matrix = math_utils.create_transformation_matrix(
                transform.position, transform.rotation, transform.scale
            )

            shader.set_mat4("u_Model", model_matrix)

            visuals.mesh.draw()

        GL.glActiveTexture(GL.GL_TEXTURE0)
