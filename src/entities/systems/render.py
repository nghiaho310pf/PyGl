from typing import Tuple

import numpy as np
from OpenGL import GL
import ctypes

from entities.components.camera_state import CameraState
from entities.components.visuals.assets import Mesh, AssetStatus
from entities.components.render_state import RenderState, GlobalDrawMode
from entities.components.camera import Camera
from entities.components.point_light import PointLight
from entities.components.directional_light import DirectionalLight
from entities.components.transform import Transform
from entities.components.visuals.visuals import Visuals, DrawMode
from entities.registry import Registry
from entities.components.visuals.material import Material
from visuals.shader import Shader, ShaderGlobals
from visuals.shaders import depth_prepass_shader, directional_shadowmap_shader, point_shadowmap_shader, shadow_blur_shader, tf2_ggx_smith, debug_depth_shader, shadow_mask_shader
import math_utils
from visuals.shaders.smaa import shaders as smaa_shaders, smaa_area_tex, smaa_search_tex


class RenderSystem:
    def __init__(self):
        # == unorthodox: global state ==
        self.shader_globals = ShaderGlobals()

        self.point_shadowmap_shader = point_shadowmap_shader.make_shader()
        self.directional_shadowmap_shader = directional_shadowmap_shader.make_shader()
        self.shadow_mask_shader = shadow_mask_shader.make_shader()
        self.shadow_blur_shader = shadow_blur_shader.make_shader()
        self.depth_prepass_shader = depth_prepass_shader.make_shader()
        self.tf2_ggx_shader = tf2_ggx_smith.make_shader()
        self.debug_depth_shader = debug_depth_shader.make_shader()
        (
            self.smaa_edge_shader,
            self.smaa_weight_shader,
            self.smaa_blend_shader
        ) = smaa_shaders.make_shaders()

        self._attach_shader(self.point_shadowmap_shader)
        self._attach_shader(self.directional_shadowmap_shader)
        self._attach_shader(self.shadow_mask_shader)
        self._attach_shader(self.shadow_blur_shader)
        self._attach_shader(self.depth_prepass_shader)
        self._attach_shader(self.tf2_ggx_shader)
        self._attach_shader(self.debug_depth_shader)
        self._attach_shader(self.smaa_edge_shader)
        self._attach_shader(self.smaa_weight_shader)
        self._attach_shader(self.smaa_blend_shader)

        self.smaa_area_tex = smaa_area_tex.load()
        self.smaa_search_tex = smaa_search_tex.load()

        self.directional_shadow_map_width = 2048
        self.directional_shadow_map_height = 2048
        self.point_shadow_map_width = 512
        self.point_shadow_map_height = 512

        # == default texture for unavailable textures ==
        self.default_texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.default_texture_id)
        white_pixel = np.array([255, 255, 255, 255], dtype=np.uint8)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, 1, 1, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, white_pixel
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # == shadow mask FBOs ==
        self.shadow_mask_fbo = 0
        self.shadow_mask_textures = []  # [point_mask, dir_mask]
        self.shadow_mask_depth = 0

        self.blur_fbo = 0
        self.blur_textures = []

        self.fbo_size = (0, 0)

    def _setup_fbos(self, width, height):
        if self.fbo_size == (width, height):
            return

        self.fbo_size = (width, height)

        if self.shadow_mask_fbo:
            GL.glDeleteFramebuffers(1, [self.shadow_mask_fbo])
            GL.glDeleteTextures(len(self.shadow_mask_textures), self.shadow_mask_textures)
            GL.glDeleteTextures(1, [self.shadow_mask_depth])

            GL.glDeleteFramebuffers(1, [self.blur_fbo])
            GL.glDeleteTextures(len(self.blur_textures), self.blur_textures)

        self.main_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.main_fbo)

        self.main_color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.main_color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, width, height, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.main_color_tex, 0)

        self.main_depth_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.main_depth_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT24, width, height, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.main_depth_tex, 0)

        self.main_normal_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.main_normal_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D, self.main_normal_tex, 0)

        # == shadow mask FBO ==
        self.shadow_mask_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.shadow_mask_fbo)

        self.shadow_mask_textures = GL.glGenTextures(3)
        for i in range(3):
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_mask_textures[i])

            internal_format = GL.GL_RGBA16F if i == 2 else GL.GL_RGBA8
            type_enum = GL.GL_FLOAT if i == 2 else GL.GL_UNSIGNED_BYTE

            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_format, width, height, 0, GL.GL_RGBA, type_enum, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0 + i, GL.GL_TEXTURE_2D, self.shadow_mask_textures[i], 0)  # type: ignore

        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.main_depth_tex, 0)
        GL.glDrawBuffers(3, (GL.GLenum * 3)(GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2))

        # == shadow blur FBO ==
        self.blur_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.blur_fbo)

        self.blur_textures = GL.glGenTextures(2)
        for i in range(2):
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.blur_textures[i])
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0 + i, GL.GL_TEXTURE_2D, self.blur_textures[i], 0)  # type: ignore

        GL.glDrawBuffers(3, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2])

        # == smaa edge FBO ==
        self.smaa_edge_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.smaa_edge_fbo)
        self.smaa_edge_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.smaa_edge_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RG8, width, height, 0, GL.GL_RG, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.smaa_edge_tex, 0)

        # == smaa weight FBO ==
        self.smaa_weight_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.smaa_weight_fbo)
        self.smaa_weight_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.smaa_weight_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.smaa_weight_tex, 0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _draw_fullscreen_quad(self):
        if not hasattr(self, "_quad_vao"):
            quad_data = np.array([
                -1.0,  1.0, 0.0, 1.0,
                -1.0, -1.0, 0.0, 0.0,
                 1.0,  1.0, 1.0, 1.0,
                 1.0, -1.0, 1.0, 0.0,
            ], dtype=np.float32)

            self._quad_vao = GL.glGenVertexArrays(1)
            self._quad_vbo = GL.glGenBuffers(1)
            GL.glBindVertexArray(self._quad_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._quad_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, quad_data.nbytes, quad_data, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(8))
            GL.glBindVertexArray(0)

        GL.glBindVertexArray(self._quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
    
    def _draw_mesh(self, mesh: Mesh):
        if mesh.status != AssetStatus.Ready:
            return

        GL.glBindVertexArray(mesh.vao)
        if mesh.has_indices:
            GL.glDrawElements(GL.GL_TRIANGLES, mesh.indices_count, GL.GL_UNSIGNED_INT, None)
        else:
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, mesh.vertex_count)
        GL.glBindVertexArray(0)

    def _attach_shader(self, shader: Shader):
        self.shader_globals.attach_to(shader)

    def _setup_directional_shadow_map(self, dir_light: DirectionalLight):
        dir_light.shadow_map_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dir_light.shadow_map_fbo)

        dir_light.shadow_map_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, dir_light.shadow_map_texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT,
                        self.directional_shadow_map_width, self.directional_shadow_map_height, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        border_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        GL.glTexParameterfv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR, border_color)

        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, dir_light.shadow_map_texture, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _setup_point_shadow_map(self, point_light: PointLight):
        point_light.shadow_map_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, point_light.shadow_map_fbo)

        point_light.shadow_map_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, point_light.shadow_map_texture)
        for i in range(6):
            GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL.GL_DEPTH_COMPONENT, # type: ignore
                            self.point_shadow_map_width, self.point_shadow_map_height, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)

        GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, point_light.shadow_map_texture, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def update(self, registry: Registry, window_size: tuple[int, int], time: float, delta_time: float):
        width, height = window_size
        self._setup_fbos(width, height)

        # == camera state setup ==
        r_admin = registry.get_singleton(RenderState)
        if r_admin is None:
            raise RuntimeError("RenderSystem is missing a RenderState singleton")
        admin_entity, (render_state, ) = r_admin

        r_camera_state = registry.get_singleton(CameraState)
        if r_camera_state is None:
            raise RuntimeError("RenderSystem is missing a CameraState singleton")
        camera_state_entity, (camera_state, ) = r_camera_state

        camera_parent = registry.get_parent(camera_state_entity)
        if camera_parent is None:
            return
        r_camera = registry.get_components(camera_parent, Transform, Camera)
        if r_camera is None:
            raise RuntimeError("CameraState singleton parented to an entity without (Transform, Camera)")
        (camera_transform, camera) = r_camera

        # == lights setup ==
        active_point_lights: list[Tuple[Transform, PointLight]] = []
        for light_entity, (point_light_transform, point_light) in registry.view(Transform, PointLight):
            if point_light.enabled:
                active_point_lights.append((point_light_transform, point_light))

        active_dir_lights: list[DirectionalLight] = []
        for light_entity, (dir_light, ) in registry.view(DirectionalLight):
            if dir_light.enabled:
                active_dir_lights.append(dir_light)

        MAX_LIGHTS = 4  # should match the shader's MAX_LIGHTS
        if len(active_point_lights) > MAX_LIGHTS:
            active_point_lights = active_point_lights[:MAX_LIGHTS]
        if len(active_dir_lights) > MAX_LIGHTS:
            active_dir_lights = active_dir_lights[:MAX_LIGHTS]

        point_light_positions = []
        point_light_colors = []
        point_light_radii = []
        point_light_far_planes = []
        point_shadow_map_textures = []
        point_light_casts_shadow = []

        for transform, point_light in active_point_lights:
            if point_light.shadow_map_fbo == 0:
                self._setup_point_shadow_map(point_light)
            point_light_positions.append(transform.position)
            point_light_colors.append(point_light.color * point_light.strength)
            point_light_radii.append(point_light.radius)
            point_light_far_planes.append(100.0)
            point_shadow_map_textures.append(point_light.shadow_map_texture)
            point_light_casts_shadow.append(1 if point_light.casts_shadow else 0)

        dir_light_directions = []
        dir_light_colors = []
        dir_shadow_map_textures = []
        dir_light_space_matrices = []
        dir_light_casts_shadow = []

        for dir_light in active_dir_lights:
            if dir_light.shadow_map_fbo == 0:
                self._setup_directional_shadow_map(dir_light)
            direction = math_utils.calculate_direction_from_rotation(dir_light.rotation)
            dir_light_directions.append(direction)
            dir_light_colors.append(dir_light.color * dir_light.strength)
            dir_shadow_map_textures.append(dir_light.shadow_map_texture)
            dir_light_casts_shadow.append(1 if dir_light.casts_shadow else 0)

        num_point_lights = len(point_light_positions)
        num_dir_lights = len(dir_light_directions)

        # == shadow map pass ==
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)

        # for point lights
        GL.glViewport(0, 0, self.point_shadow_map_width, self.point_shadow_map_height)
        point_light_projection = math_utils.create_perspective_projection(90.0, 1.0, 0.1, 100.0)
        shadow_dirs = [
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
            (np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
            (np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])),
            (np.array([0.0, 0.0, -1.0]), np.array([0.0, -1.0, 0.0])),
        ]

        for transform, point_light in active_point_lights:
            if not point_light.casts_shadow: continue
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, point_light.shadow_map_fbo)
            self.point_shadowmap_shader.use()
            self.point_shadowmap_shader.set_float("u_FarPlane", 100.0)
            self.point_shadowmap_shader.set_vec3("u_LightPos", transform.position)
            for i, (target_dir, up_dir) in enumerate(shadow_dirs):
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, point_light.shadow_map_texture, 0) # type: ignore
                GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
                light_view = math_utils.create_look_at(transform.position, transform.position + target_dir, up_dir)
                self.point_shadowmap_shader.set_mat4("u_Projection", point_light_projection)
                self.point_shadowmap_shader.set_mat4("u_View", light_view)
                for entity, (mesh_transform, visuals) in registry.view(Transform, Visuals):
                    if not visuals.enabled: continue
                    model_matrix = math_utils.create_transformation_matrix(mesh_transform.position, mesh_transform.rotation, mesh_transform.scale)
                    self.point_shadowmap_shader.set_mat4("u_Model", model_matrix)
                    self._draw_mesh(visuals.mesh)

        # for directional lights
        GL.glViewport(0, 0, self.directional_shadow_map_width, self.directional_shadow_map_height)
        dir_light_projection = math_utils.create_orthographic_projection(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0)
        for dir_light in active_dir_lights:
            if not dir_light.casts_shadow: continue
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dir_light.shadow_map_fbo)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            self.directional_shadowmap_shader.use()

            # we need a "position" for the directional light to create a view matrix.
            # since it's position-less, we just pick a spot far enough in the opposite direction.
            direction = math_utils.calculate_direction_from_rotation(dir_light.rotation)
            light_pos = -math_utils.normalize(direction) * 20.0
            light_view = math_utils.create_look_at(light_pos, np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32))

            # NOTE: multiplication order is reversed for column-major interpretation from row-major numpy
            dir_light.light_space_matrix = light_view @ dir_light_projection
            dir_light_space_matrices.append(dir_light.light_space_matrix)
            self.directional_shadowmap_shader.set_mat4("u_LightSpaceMatrix", dir_light.light_space_matrix)
            for entity, (mesh_transform, visuals) in registry.view(Transform, Visuals):
                if not visuals.enabled: continue
                model_matrix = math_utils.create_transformation_matrix(mesh_transform.position, mesh_transform.rotation, mesh_transform.scale)
                self.directional_shadowmap_shader.set_mat4("u_Model", model_matrix)
                self._draw_mesh(visuals.mesh)

        # == global state update ==
        self.shader_globals.update(camera_state.projection_matrix, camera_state.view_matrix, camera_transform.position, time)

        # == batching ==
        batches: dict[tuple[DrawMode, bool], dict[Material, list[Tuple[Transform, Visuals]]]] = {}
        for entity, (transform, visuals) in registry.view(Transform, Visuals):
            if not visuals.enabled: continue
            mat = visuals.material
            actual_draw_mode = DrawMode.Wireframe if render_state.global_draw_mode == GlobalDrawMode.Wireframe else visuals.draw_mode
            batch_key = (actual_draw_mode, visuals.cull_back_faces)
            if batch_key not in batches: batches[batch_key] = {}
            if mat not in batches[batch_key]: batches[batch_key][mat] = []
            batches[batch_key][mat].append((transform, visuals))
        sorted_batch_keys = sorted(batches.keys(), key=lambda k: (k[0].name, k[1]))

        # == depth & normal prepass ==
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.main_fbo)
        GL.glViewport(0, 0, width, height)
        GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT1])
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_COLOR_BUFFER_BIT)  # type: ignore

        self.depth_prepass_shader.use()
        for batch_key in sorted_batch_keys:
            draw_mode, cull_faces = batch_key
            if cull_faces: GL.glEnable(GL.GL_CULL_FACE); GL.glCullFace(GL.GL_BACK)
            else: GL.glDisable(GL.GL_CULL_FACE)
            for material, entities in batches[batch_key].items():
                for transform, visuals in entities:
                    model_matrix = math_utils.create_transformation_matrix(transform.position, transform.rotation, transform.scale)
                    self.depth_prepass_shader.set_mat4("u_Model", model_matrix)
                    self._draw_mesh(visuals.mesh)

        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)

        # == shadow mask pass ==
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.shadow_mask_fbo)
        GL.glDrawBuffers(2, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1])
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glDisable(GL.GL_DEPTH_TEST)

        self.shadow_mask_shader.use()
        inv_view_proj = np.linalg.inv(camera_state.view_projection_matrix)
        self.shadow_mask_shader.set_mat4("u_InverseViewProjection", inv_view_proj)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.main_depth_tex)
        self.shadow_mask_shader.set_int("u_DepthTexture", 0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.main_normal_tex)
        self.shadow_mask_shader.set_int("u_NormalTexture", 1)

        self.shadow_mask_shader.set_vec3_array("u_LightPos", point_light_positions)
        self.shadow_mask_shader.set_int("u_NumLights", num_point_lights)
        self.shadow_mask_shader.set_int_array("u_PointLightCastsShadow", point_light_casts_shadow)
        self.shadow_mask_shader.set_float_array("u_FarPlane", point_light_far_planes)
        self.shadow_mask_shader.set_float_array("u_LightRadius", point_light_radii)
        self.shadow_mask_shader.set_vec3_array("u_DirLightDirection", dir_light_directions)
        self.shadow_mask_shader.set_int("u_NumDirLights", num_dir_lights)
        self.shadow_mask_shader.set_int_array("u_DirLightCastsShadow", dir_light_casts_shadow)
        self.shadow_mask_shader.set_mat4_array("u_DirLightSpaceMatrix", dir_light_space_matrices)

        for i in range(MAX_LIGHTS):
            GL.glActiveTexture(GL.GL_TEXTURE2 + i) # type: ignore
            if i < num_point_lights: GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, point_shadow_map_textures[i])
            else: GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
        self.shadow_mask_shader.set_int_array("u_ShadowMap", list(range(2, MAX_LIGHTS + 2)))

        for i in range(MAX_LIGHTS):
            GL.glActiveTexture(GL.GL_TEXTURE2 + MAX_LIGHTS + i) # type: ignore
            if i < num_dir_lights: GL.glBindTexture(GL.GL_TEXTURE_2D, dir_shadow_map_textures[i])
            else: GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self.shadow_mask_shader.set_int_array("u_DirShadowMap", list(range(MAX_LIGHTS + 2, 2 * MAX_LIGHTS + 2)))

        self._draw_fullscreen_quad()

        # restore wireframe state for next pass
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # == shadow map blur pass ==
        GL.glDisable(GL.GL_DEPTH_TEST)
        self.shadow_blur_shader.use()
        self.shadow_blur_shader.set_float("u_Near", camera.near)
        self.shadow_blur_shader.set_float("u_Far", camera.far)
        self.shadow_blur_shader.set_float("u_DepthSensitivity", render_state.shadow_blur_depth_sensitivity)
        self.shadow_blur_shader.set_float("u_NormalThreshold", render_state.shadow_blur_normal_threshold)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.main_depth_tex)
        self.shadow_blur_shader.set_int("u_DepthTexture", 1)

        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.main_normal_tex)
        self.shadow_blur_shader.set_int("u_NormalTexture", 2)

        for i in range(2):
            # horizontal blur
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.blur_fbo)
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0 + i)  # type: ignore
            self.shadow_blur_shader.set_int("u_Horizontal", 1)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_mask_textures[i])
            self.shadow_blur_shader.set_int("u_Texture", 0)
            self._draw_fullscreen_quad()

            # vertical blur (back to main shadow mask FBO)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.shadow_mask_fbo)
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0 + i)  # type: ignore
            self.shadow_blur_shader.set_int("u_Horizontal", 0)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.blur_textures[i])
            self.shadow_blur_shader.set_int("u_Texture", 0)
            self._draw_fullscreen_quad()

        # == main pass ==
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.main_fbo)
        GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.004, 0.004, 0.004, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)  # type: ignore

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthMask(GL.GL_FALSE)

        current_shader = self.debug_depth_shader if render_state.global_draw_mode == GlobalDrawMode.DepthOnly else self.tf2_ggx_shader
        current_shader.use()

        if render_state.global_draw_mode == GlobalDrawMode.DepthOnly:
            current_shader.set_float("u_Near", camera.near)
            current_shader.set_float("u_Far", camera.far)
        else:
            current_shader.set_vec3_array("u_LightPos", point_light_positions)
            current_shader.set_vec3_array("u_LightColor", point_light_colors)
            current_shader.set_int("u_NumLights", num_point_lights)
            current_shader.set_int_array("u_PointLightCastsShadow", point_light_casts_shadow)
            current_shader.set_float_array("u_FarPlane", point_light_far_planes)
            current_shader.set_float_array("u_LightRadius", point_light_radii)
            current_shader.set_vec3_array("u_DirLightDirection", dir_light_directions)
            current_shader.set_vec3_array("u_DirLightColor", dir_light_colors)
            current_shader.set_int("u_NumDirLights", num_dir_lights)
            current_shader.set_int_array("u_DirLightCastsShadow", dir_light_casts_shadow)
            current_shader.set_mat4_array("u_DirLightSpaceMatrix", dir_light_space_matrices)
            current_shader.set_vec2("u_ScreenSize", np.array([width, height], dtype=np.float32))

            GL.glActiveTexture(GL.GL_TEXTURE8)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_mask_textures[0])
            current_shader.set_int("u_PointShadowMask", 8)
            GL.glActiveTexture(GL.GL_TEXTURE9)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_mask_textures[1])
            current_shader.set_int("u_DirShadowMask", 9)

        for batch_key in sorted_batch_keys:
            draw_mode, cull_faces = batch_key
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if draw_mode == DrawMode.Wireframe else GL.GL_FILL)
            if cull_faces: GL.glEnable(GL.GL_CULL_FACE); GL.glCullFace(GL.GL_BACK)
            else: GL.glDisable(GL.GL_CULL_FACE)
            for material, entities in batches[batch_key].items():
                if render_state.global_draw_mode != GlobalDrawMode.DepthOnly:
                    self.setup_shader_properties(current_shader, material)
                for transform, visuals in entities:
                    model_matrix = math_utils.create_transformation_matrix(transform.position, transform.rotation, transform.scale)
                    current_shader.set_mat4("u_Model", model_matrix)
                    self._draw_mesh(visuals.mesh)

        # reset to fill for subsequent passes
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # == smaa ==
        if render_state.enable_smaa:
            smaa_rt_metrics = np.array([1.0 / width, 1.0 / height, width, height], dtype=np.float32)

            # == smaa pass 1 (edge detection) ==
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.smaa_edge_fbo)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            self.smaa_edge_shader.use()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.main_color_tex)
            self.smaa_edge_shader.set_int("u_ColorTex", 0)
            self.smaa_edge_shader.set_vec4("SMAA_RT_METRICS", smaa_rt_metrics)
            self._draw_fullscreen_quad()

            # == smaa pass 2 (blend weights) ==
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.smaa_weight_fbo)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            self.smaa_weight_shader.use()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.smaa_edge_tex)
            self.smaa_weight_shader.set_int("u_EdgeTex", 0)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.smaa_area_tex)
            self.smaa_weight_shader.set_int("u_AreaTex", 1)

            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.smaa_search_tex)
            self.smaa_weight_shader.set_int("u_SearchTex", 2)

            self.smaa_weight_shader.set_vec4("SMAA_RT_METRICS", smaa_rt_metrics)
            self._draw_fullscreen_quad()

            # == smaa pass 3 (neighborhood blending), out to screen ==
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            self.smaa_blend_shader.use()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.main_color_tex)
            self.smaa_blend_shader.set_int("u_ColorTex", 0)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.smaa_weight_tex)
            self.smaa_blend_shader.set_int("u_BlendTex", 1)

            self.smaa_blend_shader.set_vec4("SMAA_RT_METRICS", smaa_rt_metrics)
            self._draw_fullscreen_quad()
        else:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.main_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
            GL.glBlitFramebuffer(
                0, 0, width, height, 
                0, 0, width, height, 
                GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST
            )

        # == clean up GL state ==
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)

    def setup_shader_properties(self, shader: Shader, material: Material):
        shader.set_vec3("u_Albedo", material.albedo)
        shader.set_float("u_Roughness", float(material.roughness))
        shader.set_float("u_Metallic", float(material.metallic))
        shader.set_float("u_Reflectance", float(material.reflectance))
        shader.set_float("u_Translucency", float(material.translucency))
        shader.set_float("u_AO", float(material.ao))

        def bind_map(tex_attr, sampler_name, flag_name, unit):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            tex = getattr(material, tex_attr, None)
            if tex and tex.status == AssetStatus.Ready and tex.gl_id:
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex.gl_id)
                shader.set_int(sampler_name, unit)
                shader.set_int(flag_name, 1)
            else:
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.default_texture_id)
                shader.set_int(sampler_name, unit)
                shader.set_int(flag_name, 0)

        bind_map("albedo_map",   "u_AlbedoMap",   "u_UseAlbedoMap",   0)
        # bind_map("normal_map",   "u_NormalMap",   "u_UseNormalMap",   1)
        # bind_map("specular_map", "u_SpecularMap", "u_UseSpecularMap", 2)
