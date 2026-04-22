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
from entities.components.street_scene.vehicle import Vehicle
from entities.components.street_scene.building import Building
from entities.components.street_scene.environment import Environment
from entities.components.gd.optimizer_state import OptimizerState
from entities.components.visuals.assets import Mesh, AssetStatus
from entities.components.render_state import RenderState, GlobalDrawMode, BoundingBox
from entities.components.point_light import PointLight
from entities.components.directional_light import DirectionalLight
from entities.components.transform import Transform
from entities.components.visuals.visuals import Visuals, DrawMode
from entities.registry import Registry
from visuals.shader import Shader, ShaderGlobals
from visuals.shaders import flat_shader, tf2_ggx_hammon, debug_depth_shader, id_shader
import math_utils


class RenderSystem:
    def __init__(self):
        # == unorthodox: global state ==
        self.shader_globals = ShaderGlobals()

        self.tf2_ggx_shader = tf2_ggx_hammon.make_shader()
        self.flat_shader = flat_shader.make_shader()
        self.debug_depth_shader = debug_depth_shader.make_shader()
        self.id_shader = id_shader.make_shader()

        # these shaders don't use ShaderGlobals:
        self._attach_shader_globals_to(self.tf2_ggx_shader)
        self._attach_shader_globals_to(self.flat_shader)
        self._attach_shader_globals_to(self.debug_depth_shader)
        self._attach_shader_globals_to(self.id_shader)

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

        if self.fbo_size != (0, 0):
            GL.glDeleteFramebuffers(1, [self.main_fbo])
            GL.glDeleteTextures(1, [self.main_color_tex])
            GL.glDeleteTextures(1, [self.main_depth_tex])

            GL.glDeleteFramebuffers(1, [self.resolve_fbo])
            GL.glDeleteTextures(1, [self.resolve_color_tex])
            GL.glDeleteTextures(1, [self.resolve_depth_tex])

            GL.glDeleteFramebuffers(1, [self.segmentation_fbo])
            GL.glDeleteTextures(1, [self.seg_color_tex])

        self.fbo_size = (width, height)
        self.samples = 4

        self.main_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.main_fbo)

        self.main_color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.main_color_tex)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.samples, GL.GL_RGBA16F, width, height, GL.GL_TRUE)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D_MULTISAMPLE, self.main_color_tex, 0)

        self.main_depth_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.main_depth_tex)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.samples, GL.GL_DEPTH_COMPONENT24, width, height, GL.GL_TRUE)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D_MULTISAMPLE, self.main_depth_tex, 0)

        # resolve FBO for post-processing and reading
        self.resolve_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.resolve_fbo)

        self.resolve_color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.resolve_color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, width, height, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.resolve_color_tex, 0)

        self.resolve_depth_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.resolve_depth_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT24, width, height, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.resolve_depth_tex, 0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # == segmentation mask FBO ==
        self.segmentation_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.segmentation_fbo)

        self.seg_color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.seg_color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32UI, width, height, 0, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, None)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.seg_color_tex, 0)
        # reuse resolve depth texture (segmentation is not multisampled for simplicity in this renderer)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.resolve_depth_tex, 0)

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

    @staticmethod
    def _get_classification_and_color(registry: Registry, entity_id: int) -> Tuple[int, str, list[int]]:
        if registry.get_components(entity_id, Vehicle):
            return 1, "Vehicle", [255, 0, 0]
        if registry.get_components(entity_id, Building):
            return 2, "Building", [0, 255, 0]
        if registry.get_components(entity_id, Environment):
            return 0, "Environment", [0, 0, 0]
        return 0, "Environment", [0, 0, 0]  # default

    def _export_dataset_frame(self, registry: Registry, width, height, render_state: RenderState, camera_state: CameraState, frame_name: str, segmentation_ids: np.ndarray):
        base_path = Path("dataset")

        rgb_dir = base_path / "images"
        depth_dir = base_path / "depth"
        label_dir = base_path / "labels"
        seg_dir = base_path / "segmentation"

        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)

        # == rgb ==
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.resolve_fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        raw_rgb = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        if not isinstance(raw_rgb, bytes):
            raise RuntimeError("GL.glReadPixels did not return bytes for RGB buffer")
        img_rgb = Image.frombytes("RGB", (width, height), raw_rgb)
        img_rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(rgb_dir / f"{frame_name}.png")

        # == depth ==
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.resolve_fbo)
        raw_depth = GL.glReadPixels(0, 0, width, height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        if not isinstance(raw_depth, np.ndarray):
            raise RuntimeError("GL.glReadPixels did not return a NumPy array for depth buffer")
        z_n = raw_depth.reshape((height, width))
        near, far = camera_state.camera_near, camera_state.camera_far
        z_lin = (near * far) / (far - z_n * (far - near))
        depth_mm = np.clip(z_lin * 1000, 0, 65535).astype(np.uint16)

        img_depth = Image.fromarray(depth_mm, mode="I;16")
        img_depth.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(depth_dir / f"{frame_name}.png")

        # == yolo labels ==
        yolo_lines = []

        for bbox in render_state.bounding_boxes:
            x_center = (bbox.min_x + bbox.max_x) / 2.0
            y_center = (bbox.min_y + bbox.max_y) / 2.0
            w, h = (bbox.max_x - bbox.min_x), (bbox.max_y - bbox.min_y)

            class_id, _, _ = self._get_classification_and_color(registry, bbox.entity_id)
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        with open(label_dir / f"{frame_name}.txt", "w") as f:
            f.writelines(yolo_lines)

        # == segmentation map ==
        seg_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for entity_id in np.unique(segmentation_ids):
            if entity_id == 0: continue  # skip background

            _, _, color = self._get_classification_and_color(registry, int(entity_id))
            seg_rgb[segmentation_ids == entity_id] = color

        img_seg = Image.fromarray(seg_rgb, mode="RGB")
        img_seg.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(seg_dir / f"{frame_name}.png")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _calculate_bounding_boxes(self, render_state: RenderState, registry: Registry, width: int, height: int):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.segmentation_fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        raw_data = GL.glReadPixels(0, 0, width, height, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT)
        if not isinstance(raw_data, np.ndarray):
            raise RuntimeError("GL.glReadPixels did not return a NumPy array for segmentation buffer")

        id_buffer = raw_data.reshape((height, width))

        render_state.bounding_boxes.clear()

        visible_entity_ids = np.unique(id_buffer)

        for entity_id in visible_entity_ids:
            if entity_id == 0:
                continue  # background

            class_id, class_name, _ = self._get_classification_and_color(registry, int(entity_id))
            if class_id == 0:  # environment
                continue

            entity_pixel_mask = (id_buffer == entity_id)

            has_entity_pixels_in_row = np.any(entity_pixel_mask, axis=1)
            has_entity_pixels_in_column = np.any(entity_pixel_mask, axis=0)

            occupied_rows = np.where(has_entity_pixels_in_row)[0]
            occupied_columns = np.where(has_entity_pixels_in_column)[0]

            if occupied_rows.size == 0 or occupied_columns.size == 0:
                continue

            bottom_row_index, top_row_index = occupied_rows[0], occupied_rows[-1]
            left_column_index, right_column_index = occupied_columns[0], occupied_columns[-1]

            name = "Unknown"
            flags_comps = registry.get_components(int(entity_id), EntityFlags)
            if flags_comps:
                name = flags_comps[0].name

            render_state.bounding_boxes.append(BoundingBox(
                entity_id=int(entity_id),
                name=name,
                classification_name=class_name,
                min_x=float(left_column_index) / width,
                min_y=1.0 - float(top_row_index + 1) / height,
                max_x=float(right_column_index + 1) / width,
                max_y=1.0 - float(bottom_row_index) / height
            ))

        return id_buffer

    @staticmethod
    def _find_visual_children(registry: Registry, entity: int) -> list[Tuple[int, Transform, Visuals]]:
        results = []
        children = registry.get_children(entity)
        for child in children:
            comps = registry.get_components(child, Transform, Visuals)
            if comps:
                results.append((child, comps[0], comps[1]))
            results.extend(RenderSystem._find_visual_children(registry, child))
        return results

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

        for transform, point_light in active_point_lights:
            point_light_positions.append(transform.world.position)
            point_light_colors.append(point_light.color * point_light.strength)
            point_light_far_planes.append(100.0)

        dir_light_directions = []
        dir_light_colors = []

        for dir_light in active_dir_lights:
            direction = math_utils.calculate_direction_from_rotation(dir_light.rotation)
            dir_light_directions.append(direction)
            dir_light_colors.append(dir_light.color * dir_light.strength)

        num_point_lights = len(point_light_positions)
        num_dir_lights = len(dir_light_directions)

        # == batching ==
        batches: dict[tuple[DrawMode, bool], list[Tuple[Transform, Visuals]]] = {}
        for entity, (transform, visuals) in registry.view(Transform, Visuals):
            if not visuals.enabled: continue

            actual_draw_mode = DrawMode.Wireframe if render_state.global_draw_mode == GlobalDrawMode.Wireframe else visuals.draw_mode
            batch_key = (actual_draw_mode, visuals.cull_back_faces)
            if batch_key not in batches: batches[batch_key] = []
            math_utils.update_transformation_matrix(
                transform.world.position, transform.world.rotation, transform.world.scale,
                transform.matrix_cache
            )
            batches[batch_key].append((transform, visuals))
        sorted_batch_keys = sorted(batches.keys(), key=lambda k: (k[0].name, k[1]))

        # == global state update ==
        self.shader_globals.update(camera_state.projection_matrix, camera_state.view_matrix, camera_state.camera_position, time_val)

        # == main pass (multisampled) ==
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.main_fbo)
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.004, 0.004, 0.004, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # type: ignore

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_MULTISAMPLE)

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
                for transform, visuals in batches[batch_key]:
                    current_shader.set_mat4("u_Model", transform.matrix_cache)
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

            for batch_key in sorted_batch_keys:
                draw_mode, cull_faces = batch_key
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if draw_mode == DrawMode.Wireframe else GL.GL_FILL)
                if cull_faces: GL.glEnable(GL.GL_CULL_FACE); GL.glCullFace(GL.GL_BACK)
                else: GL.glDisable(GL.GL_CULL_FACE)
                for transform, visuals in batches[batch_key]:
                    self.tf2_ggx_shader.set_material(visuals.material, self.default_texture_id)
                    current_shader.set_mat4("u_Model", transform.matrix_cache)
                    self._draw_mesh(visuals.mesh)

        # == trajectory line rendering (on multisampled FBO) ==
        GL.glBindVertexArray(self.line_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.line_vbo)

        self.flat_shader.use()
        self.flat_shader.set_mat4("u_Model", np.identity(4, dtype=np.float32))
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        GL.glDepthRange(0.0, 0.9998)
        for entity, (optimizer,) in registry.view(OptimizerState):
            if len(optimizer.trajectory) < 2: continue
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
            GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(optimizer.trajectory))
        GL.glBindVertexArray(0)
        GL.glDepthRange(0.0, 1.0)

        # == resolve MSAA ==
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.main_fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.resolve_fbo)
        GL.glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height,
            GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT, GL.GL_NEAREST  # type: ignore
        )

        # == segmentation mask pass (non-multisampled) ==
        segmentation_ids = None
        if render_state.show_bounding_boxes or render_state.is_capture:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.segmentation_fbo)
            GL.glViewport(0, 0, width, height)
            GL.glClearColor(0, 0, 0, 0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # type: ignore
            GL.glDisable(GL.GL_MULTISAMPLE)
            GL.glEnable(GL.GL_DEPTH_TEST)

            self.id_shader.use()
            classified_entities = []
            for e, _ in registry.view(Vehicle): classified_entities.append(e)
            for e, _ in registry.view(Building): classified_entities.append(e)
            for e, _ in registry.view(Environment): classified_entities.append(e)

            for classified_entity in classified_entities:
                comps = registry.get_components(classified_entity, Transform, Visuals)
                if comps and comps[1].enabled:
                    self.id_shader.set_uint("u_EntityID", classified_entity)
                    model_matrix = math_utils.create_transformation_matrix(
                        comps[0].world.position, comps[0].world.rotation, comps[0].world.scale
                    )
                    self.id_shader.set_mat4("u_Model", model_matrix)
                    self._draw_mesh(comps[1].mesh)

                visual_children = RenderSystem._find_visual_children(registry, classified_entity)
                for child_id, transform, visuals in visual_children:
                    if not visuals.enabled: continue
                    self.id_shader.set_uint("u_EntityID", classified_entity)
                    model_matrix = math_utils.create_transformation_matrix(
                        transform.world.position, transform.world.rotation, transform.world.scale
                    )
                    self.id_shader.set_mat4("u_Model", model_matrix)
                    self._draw_mesh(visuals.mesh)

            segmentation_ids = self._calculate_bounding_boxes(render_state, registry, width, height)

        # == final output ==
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.resolve_fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
        GL.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)

        # == handle capture ==
        if render_state.is_capture:
            self._export_dataset_frame(registry, width, height, render_state, camera_state, str(render_state.frame_number).zfill(6), segmentation_ids)  # type: ignore
            render_state.is_capture = False

        # == performance metrics ==
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
