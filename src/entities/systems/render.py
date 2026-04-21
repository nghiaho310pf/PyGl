import colorsys
from pathlib import Path
from typing import Tuple
import ctypes

import numpy as np
import numpy.typing as npt
from OpenGL import GL
import PIL.Image as Image

from engine.application import Application
from entities.components.camera_state import CameraState
from entities.components.entity_flags import EntityFlags
from entities.components.gd.optimizer_state import OptimizerState
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
from visuals.shaders import depth_prepass_shader, directional_shadowmap_shader, flat_shader, point_shadowmap_shader, shadow_blur_shader, tf2_ggx_hammon, debug_depth_shader, id_shader
from visuals.shaders.shadow_mask_shader import blue_noise_tex
from visuals.shaders.shadow_mask_shader import shaders as shadow_mask_shader
from visuals.shaders.smaa import shaders as smaa_shaders, smaa_area_tex, smaa_search_tex
import math_utils


class RenderSystem:
    def __init__(self):
        # == unorthodox: global state ==
        self.shader_globals = ShaderGlobals()

        self.point_shadowmap_shader = point_shadowmap_shader.make_shader()
        self.directional_shadowmap_shader = directional_shadowmap_shader.make_shader()
        self.shadow_mask_shader = shadow_mask_shader.make_shader()
        self.shadow_blur_shader = shadow_blur_shader.make_shader()
        self.depth_prepass_shader = depth_prepass_shader.make_shader()
        self.tf2_ggx_shader = tf2_ggx_hammon.make_shader()
        self.flat_shader = flat_shader.make_shader()
        self.debug_depth_shader = debug_depth_shader.make_shader()
        self.id_shader = id_shader.make_shader()
        (
            self.smaa_edge_shader,
            self.smaa_weight_shader,
            self.smaa_blend_shader
        ) = smaa_shaders.make_shaders()

        # these shaders don't use ShaderGlobals:
        # - point_shadowmap_shader
        # - directional_shadowmap_shader
        # - shadow_blur_shader
        # - smaa_edge_shader
        # - smaa_weight_shader
        # - smaa_blend_shader
        self._attach_shader_globals_to(self.shadow_mask_shader)
        self._attach_shader_globals_to(self.depth_prepass_shader)
        self._attach_shader_globals_to(self.tf2_ggx_shader)
        self._attach_shader_globals_to(self.flat_shader)
        self._attach_shader_globals_to(self.debug_depth_shader)
        self._attach_shader_globals_to(self.id_shader)

        self.blue_noise_tex = blue_noise_tex.load()
        self.smaa_area_tex = smaa_area_tex.load()
        self.smaa_search_tex = smaa_search_tex.load()

        self.directional_shadow_map_width = 1408
        self.directional_shadow_map_height = 1408
        self.point_shadow_map_width = 512
        self.point_shadow_map_height = 512

        # == fullscreen quad setup ==
        self._setup_fullscreen_quad()

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

        # == dynamic line rendering setup ==
        # very unorthodox, but we have deadlines to meet
        self.line_vao = GL.glGenVertexArrays(1)
        self.line_vbo = GL.glGenBuffers(1)

        # == performance querying ==
        if not Application.has_broken_opengl:
            self.num_timer_queries = 8
            self.timer_queries = GL.glGenQueries(self.num_timer_queries)
            self.query_index = 0
            self.gpu_time_ms = 0.0

    def _setup_fullscreen_quad(self):
        quad_data = np.array([
            -1.0, -1.0, 0.0, 0.0,
             3.0, -1.0, 2.0, 0.0,
            -1.0,  3.0, 0.0, 2.0,
        ], dtype=np.float32)

        self._fullscreen_quad_vao = GL.glGenVertexArrays(1)
        self._fullscreen_quad_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._fullscreen_quad_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._fullscreen_quad_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad_data.nbytes, quad_data, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(8))
        GL.glBindVertexArray(0)

    def _draw_fullscreen_quad(self):
        GL.glBindVertexArray(self._fullscreen_quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)

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

        self.shadow_mask_textures = GL.glGenTextures(2)
        for i in range(2):
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_mask_textures[i])

            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0 + i, GL.GL_TEXTURE_2D, self.shadow_mask_textures[i], 0)  # type: ignore

        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.main_depth_tex, 0)
        GL.glDrawBuffers(2, (GL.GLenum * 2)(GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1))

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

        GL.glDrawBuffers(2, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1])

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

        # == segmentation mask FBO ==
        self.segmentation_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.segmentation_fbo)

        self.seg_color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.seg_color_tex)
        # TOOD: use GL_RGB8 for color-coded IDs or GL_R32I for actual integer IDs?
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.seg_color_tex, 0)
        # reuse existing depth texture
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.main_depth_tex, 0)

    def _draw_mesh(self, mesh: Mesh):
        if mesh.status != AssetStatus.Ready:
            return

        GL.glBindVertexArray(mesh.vao)
        if mesh.has_indices:
            GL.glDrawElements(GL.GL_TRIANGLES, mesh.indices_count, GL.GL_UNSIGNED_INT, None)
        else:
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, mesh.vertex_count)
        GL.glBindVertexArray(0)

    def _attach_shader_globals_to(self, shader: Shader):
        self.shader_globals.attach_to(shader)

    def _setup_directional_shadow_map(self, dir_light: DirectionalLight):
        dir_light.shadow_map_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dir_light.shadow_map_fbo)

        dir_light.shadow_map_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, dir_light.shadow_map_texture)
        GL.glTexImage3D(GL.GL_TEXTURE_2D_ARRAY, 0, GL.GL_DEPTH_COMPONENT,
                        self.directional_shadow_map_width, self.directional_shadow_map_height, 3, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        border_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        GL.glTexParameterfv(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_BORDER_COLOR, border_color)

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

    def _export_dataset_frame(self, registry: Registry, width, height, camera_state: CameraState, frame_name: str):
        exports = {
            "images": (0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, "RGB"),
            "segmentation": (self.segmentation_fbo, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, "RGB"),
            "depth": (self.main_fbo, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, "F"),
        }

        base_path = Path("dataset")

        for folder, (fbo, gl_fmt, gl_type, pil_mode) in exports.items():
            export_dir = base_path / folder
            export_dir.mkdir(parents=True, exist_ok=True)

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

            if fbo == self.segmentation_fbo:
                GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            elif fbo == 0:
                GL.glReadBuffer(GL.GL_BACK)

            raw_data = GL.glReadPixels(0, 0, width, height, gl_fmt, gl_type)
            save_path = export_dir / f"{frame_name}.png"

            if folder == "depth":
                if not isinstance(raw_data, np.ndarray):
                    raise RuntimeError("GL.glReadPixels did not return a NumPy array for depth buffer")
                z_n = raw_data.reshape((height, width))
                z_lin = (2.0 * camera_state.camera_near * camera_state.camera_far) / (camera_state.camera_far + camera_state.camera_near - z_n * (camera_state.camera_far - camera_state.camera_near))
                depth_mm = (z_lin * 1000).astype(np.uint16)

                img = Image.fromarray(depth_mm, mode="I;16")
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                img.save(save_path)
            else:
                if not isinstance(raw_data, bytes):
                    raise RuntimeError("GL.glReadPixels did not return bytes for non-depth buffer")
                img = Image.frombytes(pil_mode, (width, height), raw_data)
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                if folder == "segmentation":
                    self._export_yolo_labels(base_path / "labels", frame_name, width, height, np.array(img), registry)

                img.save(save_path)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _get_distributed_color(self, entity_id: int) -> np.ndarray:
        phi = 0.618033988749895
        h = (entity_id * phi) % 1.0
        s = 0.5 + (entity_id % 5) * 0.1
        v = 0.95
        return np.array(colorsys.hsv_to_rgb(h, s, v))

    def _export_yolo_labels(self, export_dir: Path, frame_name: str, width: int, height: int, seg_array: np.ndarray, registry: Registry):
        export_dir.mkdir(parents=True, exist_ok=True)
        yolo_lines = []

        for entity, (transform, visuals, flags) in registry.view(Transform, Visuals, EntityFlags):
            if not visuals.enabled: continue

            color_float = self._get_distributed_color(entity)
            target_color = np.clip(np.round(color_float * 255), 0, 255).astype(np.uint8)

            mask = (seg_array[:, :, 0] == target_color[0]) & \
                   (seg_array[:, :, 1] == target_color[1]) & \
                   (seg_array[:, :, 2] == target_color[2])

            coords = np.argwhere(mask)
            if coords.size == 0:
                # occluded, skip
                continue

            # NOTE: argwhere returns (y, x)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # normalize for YOLO [x_center, y_center, width, height]
            x_center = (x_min + x_max) / (2.0 * width)
            y_center = (y_min + y_max) / (2.0 * height)
            bbox_w = (x_max - x_min) / float(width)
            bbox_h = (y_max - y_min) / float(height)

            yolo_lines.append(f"{flags.classification} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

        with open(export_dir / f"{frame_name}.txt", "w") as f:
            f.writelines(yolo_lines)

    @staticmethod
    def _smooth_metric(current_avg: float, new_value: float) -> float:
        base_alpha = 0.01
        sensitivity = 0.5

        if current_avg == 0.0:
            return new_value

        relative_diff = abs(new_value - current_avg) / max(current_avg, 0.001)
        dynamic_alpha = min(1.0, base_alpha + (relative_diff * sensitivity))

        return (new_value * dynamic_alpha) + (current_avg * (1.0 - dynamic_alpha))

    def update(self, registry: Registry, window_size: tuple[int, int], time_val: float, delta_time: float):
        if not Application.has_broken_opengl:
            GL.glBeginQuery(GL.GL_TIME_ELAPSED, self.timer_queries[self.query_index])

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

        # choose graphics settings for this frame
        if render_state.is_capture:
            graphics_settings = render_state.capture_graphics_settings
        else:
            graphics_settings = render_state.viewport_graphics_settings

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
        point_light_far_planes = []
        point_shadow_map_textures = []
        point_light_casts_shadow = []

        for transform, point_light in active_point_lights:
            if point_light.shadow_map_fbo == 0:
                self._setup_point_shadow_map(point_light)
            point_light_positions.append(transform.world.position)
            point_light_colors.append(point_light.color * point_light.strength)
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

        # == batching ==
        batches: dict[tuple[DrawMode, bool], list[Tuple[Transform, Visuals, npt.NDArray[np.float32]]]] = {}
        for entity, (transform, visuals) in registry.view(Transform, Visuals):
            if not visuals.enabled: continue

            actual_draw_mode = DrawMode.Wireframe if render_state.global_draw_mode == GlobalDrawMode.Wireframe else visuals.draw_mode
            batch_key = (actual_draw_mode, visuals.cull_back_faces)
            if batch_key not in batches: batches[batch_key] = []
            model_matrix = math_utils.create_transformation_matrix(
                transform.world.position, transform.world.rotation, transform.world.scale
            )
            batches[batch_key].append((transform, visuals, model_matrix))
        sorted_batch_keys = sorted(batches.keys(), key=lambda k: (k[0].name, k[1]))

        # == shadow map pass ==
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_TRUE)

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
            self.point_shadowmap_shader.set_vec3("u_LightPos", transform.world.position)
            for i, (target_dir, up_dir) in enumerate(shadow_dirs):
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, point_light.shadow_map_texture, 0) # type: ignore
                GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
                light_view = math_utils.create_look_at(transform.world.position, transform.world.position + target_dir, up_dir)
                self.point_shadowmap_shader.set_mat4("u_Projection", point_light_projection)
                self.point_shadowmap_shader.set_mat4("u_View", light_view)

                for batch_key in sorted_batch_keys:
                    draw_mode, cull_faces = batch_key
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if draw_mode == DrawMode.Wireframe else GL.GL_FILL)
                    if cull_faces:
                        GL.glEnable(GL.GL_CULL_FACE)
                        GL.glCullFace(GL.GL_BACK)
                    else:
                        GL.glDisable(GL.GL_CULL_FACE)
                    for mesh_transform, visuals, model_matrix in batches[batch_key]:
                        self.point_shadowmap_shader.set_mat4("u_Model", model_matrix)
                        self._draw_mesh(visuals.mesh)

        # for directional lights
        GL.glViewport(0, 0, self.directional_shadow_map_width, self.directional_shadow_map_height)

        proj = camera_state.projection_matrix
        fovy = np.degrees(np.arctan(1.0 / proj[1, 1]) * 2.0)
        aspect = proj[1, 1] / proj[0, 0]

        cascade_levels = [
            (camera_state.camera_near, camera_state.camera_far * 0.08),
            (camera_state.camera_near, camera_state.camera_far * 0.25),
            (camera_state.camera_near, camera_state.camera_far)
        ]
        cascade_distances = [c[1] for c in cascade_levels]

        for i, dir_light in enumerate(active_dir_lights):
            dir_light.light_space_matrices.clear()
            dir_light.cascade_distances = cascade_distances

            if not dir_light.casts_shadow:
                for _ in range(3):
                    mat = np.eye(4, dtype=np.float32)
                    dir_light.light_space_matrices.append(mat)
                    dir_light_space_matrices.append(mat)
                continue

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dir_light.shadow_map_fbo)
            direction = dir_light_directions[i]

            for cascade_idx, (c_near, c_far) in enumerate(cascade_levels):
                GL.glFramebufferTextureLayer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, dir_light.shadow_map_texture, 0, cascade_idx)
                GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
                self.directional_shadowmap_shader.use()

                cascade_proj = math_utils.create_perspective_projection(fovy, aspect, c_near, c_far)
                light_space_mat = math_utils.get_light_space_matrix(cascade_proj, camera_state.view_matrix, direction)

                dir_light.light_space_matrices.append(light_space_mat)
                dir_light_space_matrices.append(light_space_mat)
                self.directional_shadowmap_shader.set_mat4("u_LightSpaceMatrix", light_space_mat)

                for batch_key in sorted_batch_keys:
                    draw_mode, cull_faces = batch_key
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if draw_mode == DrawMode.Wireframe else GL.GL_FILL)
                    if cull_faces:
                        GL.glEnable(GL.GL_CULL_FACE)
                        GL.glCullFace(GL.GL_BACK)
                    else:
                        GL.glDisable(GL.GL_CULL_FACE)
                    for mesh_transform, visuals, model_matrix in batches[batch_key]:
                        self.directional_shadowmap_shader.set_mat4("u_Model", model_matrix)
                        self._draw_mesh(visuals.mesh)

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # == global state update ==
        self.shader_globals.update(camera_state.projection_matrix, camera_state.view_matrix, camera_state.camera_position, time_val)

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
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if draw_mode == DrawMode.Wireframe else GL.GL_FILL)
            if cull_faces: GL.glEnable(GL.GL_CULL_FACE); GL.glCullFace(GL.GL_BACK)
            else: GL.glDisable(GL.GL_CULL_FACE)
            for transform, visuals, model_matrix in batches[batch_key]:
                self.depth_prepass_shader.set_mat4("u_Model", model_matrix)
                self._draw_mesh(visuals.mesh)

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
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

        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.blue_noise_tex)
        self.shadow_mask_shader.set_int("u_BlueNoiseTexture", 2)

        self.shadow_mask_shader.set_int("u_PointPcfSamples", graphics_settings.point_shadow_samples)
        self.shadow_mask_shader.set_float("u_InvPointPcfSamples", 1.0 / graphics_settings.point_shadow_samples)
        self.shadow_mask_shader.set_float("u_InvSqrtPointPcfSamples", graphics_settings.point_shadow_samples ** -0.5)

        self.shadow_mask_shader.set_int("u_DirPcfSamples", graphics_settings.directional_shadow_samples)
        self.shadow_mask_shader.set_float("u_InvDirPcfSamples", 1.0 / graphics_settings.directional_shadow_samples)
        self.shadow_mask_shader.set_float("u_InvSqrtDirPcfSamples", graphics_settings.directional_shadow_samples ** -0.5)

        self.shadow_mask_shader.set_vec3_array("u_LightPos", point_light_positions)
        self.shadow_mask_shader.set_int("u_NumLights", num_point_lights)
        self.shadow_mask_shader.set_int_array("u_PointLightCastsShadow", point_light_casts_shadow)
        self.shadow_mask_shader.set_float_array("u_FarPlane", point_light_far_planes)
        self.shadow_mask_shader.set_vec3_array("u_DirLightDirection", dir_light_directions)
        self.shadow_mask_shader.set_int("u_NumDirLights", num_dir_lights)
        self.shadow_mask_shader.set_int_array("u_DirLightCastsShadow", dir_light_casts_shadow)
        self.shadow_mask_shader.set_mat4_array("u_DirLightSpaceMatrices", dir_light_space_matrices)
        self.shadow_mask_shader.set_float_array("u_CascadeDistances", cascade_distances)

        for i in range(MAX_LIGHTS):
            GL.glActiveTexture(GL.GL_TEXTURE3 + i) # type: ignore
            if i < num_point_lights: GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, point_shadow_map_textures[i])
            else: GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
        self.shadow_mask_shader.set_int_array("u_ShadowMap", list(range(3, MAX_LIGHTS + 3)))

        for i in range(MAX_LIGHTS):
            GL.glActiveTexture(GL.GL_TEXTURE3 + MAX_LIGHTS + i) # type: ignore
            if i < num_dir_lights: GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, dir_shadow_map_textures[i])
            else: GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)
        self.shadow_mask_shader.set_int_array("u_DirShadowMap", list(range(MAX_LIGHTS + 3, 2 * MAX_LIGHTS + 3)))

        self._draw_fullscreen_quad()

        # restore wireframe state for next pass
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # == shadow map blur pass ==
        depth_params = math_utils.vec2(
            camera_state.camera_far + camera_state.camera_near,
            camera_state.camera_far - camera_state.camera_near
        ) / (2.0 * camera_state.camera_near * camera_state.camera_far)

        GL.glDisable(GL.GL_DEPTH_TEST)
        self.shadow_blur_shader.use()
        self.shadow_blur_shader.set_vec2("u_DepthParams", depth_params)
        self.shadow_blur_shader.set_float("u_DepthSensitivity", render_state.shadow_blur_depth_sensitivity)
        self.shadow_blur_shader.set_float("u_NormalThreshold", render_state.shadow_blur_normal_threshold)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.main_depth_tex)
        self.shadow_blur_shader.set_int("u_DepthTexture", 1)

        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.main_normal_tex)
        self.shadow_blur_shader.set_int("u_NormalTexture", 2)

        for i in range(2):
            if not any(point_light_casts_shadow if i == 0 else dir_light_casts_shadow):
                continue

            # horizontal blur
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.blur_fbo)
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0 + i)  # type: ignore
            self.shadow_blur_shader.set_vec2("u_TexelOffset", (1.0 / width, 0.0))
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_mask_textures[i])
            self.shadow_blur_shader.set_int("u_Texture", 0)
            self._draw_fullscreen_quad()

            # vertical blur (back to main shadow mask FBO)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.shadow_mask_fbo)
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0 + i)  # type: ignore
            self.shadow_blur_shader.set_vec2("u_TexelOffset", (0.0, 1.0 / height))
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
            current_shader.set_float("u_Near", camera_state.camera_near)
            current_shader.set_float("u_Far", camera_state.camera_far)

            for batch_key in sorted_batch_keys:
                draw_mode, cull_faces = batch_key
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if draw_mode == DrawMode.Wireframe else GL.GL_FILL)
                if cull_faces: GL.glEnable(GL.GL_CULL_FACE); GL.glCullFace(GL.GL_BACK)
                else: GL.glDisable(GL.GL_CULL_FACE)
                for transform, visuals, model_matrix in batches[batch_key]:
                    current_shader.set_mat4("u_Model", model_matrix)
                    self._draw_mesh(visuals.mesh)
        else:
            current_shader.set_vec3_array("u_LightPos", point_light_positions)
            current_shader.set_vec3_array("u_LightColor", point_light_colors)
            current_shader.set_float_array("u_LightFarPlane", point_light_far_planes)
            current_shader.set_int("u_NumLights", num_point_lights)
            current_shader.set_vec3_array("u_DirLightDirection", dir_light_directions)
            current_shader.set_vec3_array("u_DirLightColor", dir_light_colors)
            current_shader.set_int("u_NumDirLights", num_dir_lights)
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
                for transform, visuals, model_matrix in batches[batch_key]:
                    self.tf2_ggx_shader.set_material(visuals.material, self.default_texture_id)
                    current_shader.set_mat4("u_Model", model_matrix)
                    self._draw_mesh(visuals.mesh)

        # reset to fill for subsequent passes
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # == segmentation mask pass ==
        if render_state.is_capture:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.segmentation_fbo)
            GL.glViewport(0, 0, width, height)
            GL.glClearColor(0, 0, 0, 0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            self.id_shader.use()

            for entity, (transform, visuals) in registry.view(Transform, Visuals):
                if not visuals.enabled: continue

                target_color = self._get_distributed_color(entity)
                self.id_shader.set_vec3("u_EntityColor", target_color)

                # i don't particularly care about optimizing this create_transformation_matrix call away
                model_matrix = math_utils.create_transformation_matrix(
                    transform.world.position, transform.world.rotation, transform.world.scale
                )
                self.id_shader.set_mat4("u_Model", model_matrix)
                self._draw_mesh(visuals.mesh)

        # == trajectory line rendering ==
        GL.glBindVertexArray(self.line_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.line_vbo)

        self.flat_shader.use()
        self.flat_shader.set_mat4("u_Model", np.identity(4, dtype=np.float32))
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        GL.glDepthRange(0.0, 0.9998)
        for entity, (optimizer,) in registry.view(OptimizerState):
            if len(optimizer.trajectory) < 2:
                continue

            r_visuals = registry.get_components(entity, Visuals)
            if r_visuals:
                (vis, ) = r_visuals
                self.flat_shader.set_vec3("u_Albedo", vis.material.albedo)
            else:
                self.flat_shader.set_vec3("u_Albedo", np.array([1.0, 1.0, 1.0], dtype=np.float32))

            points = np.array(optimizer.trajectory, dtype=np.float32).flatten()
            GL.glBufferData(GL.GL_ARRAY_BUFFER, points.nbytes, points, GL.GL_DYNAMIC_DRAW)

            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 3 * 4, None)
            GL.glDisableVertexAttribArray(1)
            GL.glDisableVertexAttribArray(2)

            GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(optimizer.trajectory))
        GL.glBindVertexArray(0)
        GL.glDepthRange(0.0, 1.0)

        # == smaa ==
        if graphics_settings.enable_smaa:
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

        # == handle capture ==
        if render_state.is_capture:
            self._export_dataset_frame(registry, width, height, camera_state, str(render_state.frame_number).zfill(6))
            render_state.is_capture = False

        if not Application.has_broken_opengl:
            GL.glEndQuery(GL.GL_TIME_ELAPSED)

            if render_state.frame_number >= self.num_timer_queries - 1:
                oldest_query_index = (self.query_index + 1) % self.num_timer_queries
                available = GL.glGetQueryObjectiv(self.timer_queries[oldest_query_index], GL.GL_QUERY_RESULT_AVAILABLE)
                if available:
                    elapsed_ns = GL.glGetQueryObjectuiv(self.timer_queries[oldest_query_index], GL.GL_QUERY_RESULT)
                    self.gpu_time_ms = elapsed_ns / 1_000_000.0

            self.query_index = (self.query_index + 1) % self.num_timer_queries

            current_fps = 1.0 / delta_time if delta_time > 0 else 0.0
            current_theoretical_fps = 1000.0 / self.gpu_time_ms if self.gpu_time_ms > 0 else 0.0

            render_state.render_time_ms = RenderSystem._smooth_metric(render_state.render_time_ms, self.gpu_time_ms)
            render_state.fps = RenderSystem._smooth_metric(render_state.fps, current_fps)
            render_state.theoretical_max_fps = RenderSystem._smooth_metric(render_state.theoretical_max_fps, current_theoretical_fps)

        render_state.frame_number += 1
