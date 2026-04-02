from typing import Tuple

import numpy as np
from OpenGL import GL

from entities.components.camera_state import CameraState
from entities.components.textures_state import TextureStatus
from entities.components.render_state import RenderState, GlobalDrawMode
from entities.components.camera import Camera
from entities.components.point_light import PointLight
from entities.components.transform import Transform
from entities.components.visuals import Visuals, DrawMode
from entities.registry import Registry
from shading.material import Material
from shading.shader import Shader, ShaderGlobals
from shading.shaders import blinn_phong, depth_shader
import math_utils


class RenderSystem:
    def __init__(self):
        # == unorthodox: global state ==
        self.shader_globals = ShaderGlobals()

        self.blinn_phong_shader = blinn_phong.make_shader()
        self.depth_shader = depth_shader.make_shader()
        self.attach_shader(self.blinn_phong_shader)
        self.attach_shader(self.depth_shader)

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

    def attach_shader(self, shader: Shader):
        self.shader_globals.attach_to(shader)

    def update(self, registry: Registry, window_size: tuple[int, int], time: float, delta_time: float):
        # == global pre-render setup ==
        width, height = window_size
        aspect_ratio = width / height if height > 0 else 1.0

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.004, 0.004, 0.004, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT) # type: ignore
        GL.glEnable(GL.GL_DEPTH_TEST)

        # == camera setup ==
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

        self.shader_globals.update(camera_state.projection_matrix,
                                   camera_state.view_matrix, camera_transform.position, time)

        # == batching ==
        batches: dict[tuple[DrawMode, bool], dict[Material, list[Tuple[Transform, Visuals]]]] = {}

        for entity, (transform, visuals) in registry.view(Transform, Visuals):
            if not visuals.enabled:
                continue

            mat = visuals.material

            if render_state.global_draw_mode == GlobalDrawMode.Wireframe:
                actual_draw_mode = DrawMode.Wireframe
            else:
                actual_draw_mode = visuals.draw_mode

            cull_faces = visuals.cull_back_faces

            batch_key = (actual_draw_mode, cull_faces)

            if batch_key not in batches:
                batches[batch_key] = {}
            if mat not in batches[batch_key]:
                batches[batch_key][mat] = []

            batches[batch_key][mat].append((transform, visuals))

        # == render ==
        sorted_keys = sorted(batches.keys(), key=lambda k: (k[0].name, k[1]))

        current_shader = None
        current_draw_mode = None
        current_cull_faces = None

        for batch_key in sorted_keys:
            draw_mode, cull_faces = batch_key
            material_group = batches[batch_key]

            if render_state.global_draw_mode == GlobalDrawMode.DepthOnly:
                shader = self.depth_shader
            else:
                shader = self.blinn_phong_shader

            # 'shader != current_shader' would suffice, Pylance is just bad
            if current_shader is None or shader != current_shader:
                shader.use()
                if render_state.global_draw_mode == GlobalDrawMode.DepthOnly:
                    shader.set_float("u_Near", camera.near)
                    shader.set_float("u_Far", camera.far)
                else:
                    shader.set_vec3_array("u_LightPos", point_light_positions)
                    shader.set_vec3_array("u_LightColor", point_light_colors)
                    shader.set_int("u_NumLights", num_lights)
                current_shader = shader

            if draw_mode != current_draw_mode:
                if draw_mode == DrawMode.Wireframe:
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
                else:
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
                current_draw_mode = draw_mode

            if cull_faces != current_cull_faces:
                if cull_faces:
                    GL.glEnable(GL.GL_CULL_FACE)
                    GL.glCullFace(GL.GL_BACK)
                else:
                    GL.glDisable(GL.GL_CULL_FACE)
                current_cull_faces = cull_faces

            for material, entities in material_group.items():
                if render_state.global_draw_mode != GlobalDrawMode.DepthOnly:
                    self.setup_shader_properties(current_shader, material)

                for transform, visuals in entities:
                    model_matrix = math_utils.create_transformation_matrix(
                        transform.position, transform.rotation, transform.scale
                    )
                    current_shader.set_mat4("u_Model", model_matrix)

                    visuals.mesh.draw()

    def setup_shader_properties(self, shader: Shader, material: Material):
        shader.set_vec3("u_Albedo", material.albedo)
        shader.set_float("u_Roughness", float(material.roughness))
        shader.set_float("u_Reflectance", float(material.reflectance))
        shader.set_float("u_AO", float(material.ao))

        GL.glActiveTexture(GL.GL_TEXTURE0)

        if (
            material.albedo_map is not None and
            material.albedo_map.status == TextureStatus.Ready and
            material.albedo_map.gl_id is not None
        ):
            GL.glBindTexture(GL.GL_TEXTURE_2D, material.albedo_map.gl_id)
            shader.set_int("u_AlbedoMap", 0)
            shader.set_int("u_UseAlbedoMap", 1)
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.default_texture_id)
            shader.set_int("u_AlbedoMap", 0)
            shader.set_int("u_UseAlbedoMap", 0)
