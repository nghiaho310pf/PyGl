import struct

import OpenGL.GL as GL

from engine.application import Application
from entities.camera import Camera
from entities.entity import Entity
from geometry.geometry import generate_sphere, generate_cube_flat
from geometry.mesh import Mesh
from shading import ggx
from shading.material import Material
from shading.shader import Shader
from shading.ubo import UniformBuffer


class RotatingEntity(Entity):
    def update(self, dt: float):
        self.rotation[1] += 1.5


class Renderer(Application):
    def __init__(self, width, height):
        super().__init__(width, height)

        GL.glFrontFace(GL.GL_CCW)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_MULTISAMPLE)

        self.camera = Camera(position=[0.0, 0.0, 2.5], aspect_ratio=width / height)

        shader = ggx.make_shader()
        self.scene_ubo = UniformBuffer(size=144, binding_point=0)
        self._bind_shader_block(shader, "SceneData", 0)

        mat_orange = Material(shader, {
            "u_Albedo": [1.0, 0.5, 0.2],
            "u_Metallic": 0.3,
            "u_Roughness": 0.3
        })
        mat_blue = Material(shader, {
            "u_Albedo": [0.459, 0.651, 1.0],
            "u_Metallic": 0.7,
            "u_Roughness": 0.3
        })

        # vertices = np.array([
        #     -0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0, # bottom left
        #      0.5, -0.5,  0.0,  0.0,  0.0,  1.0,  1.0,  0.0, # bottom right
        #      0.0,  0.5,  0.0,  0.0,  0.0,  1.0,  0.5,  1.0  # top center
        # ], dtype=np.float32)
        sphere_vertices, sphere_indices = generate_sphere(radius=0.5, stacks=32, sectors=32)
        cube_vertices, cube_indices = generate_cube_flat(size=1.0)

        self.entities = [
            Entity(Mesh(sphere_vertices, sphere_indices), mat_orange, position=(-0.7, 0, 0)),
            RotatingEntity(Mesh(cube_vertices, cube_indices), mat_blue, position=(0.7, 0, 0))
        ]

        self.light_pos = [2.0, 2.0, 5.0]
        self.light_color = [300.0, 300.0, 300.0]

        self.last_update = None

    def _bind_shader_block(self, shader: Shader, block_name: str, binding_point: int):
        block_index = GL.glGetUniformBlockIndex(shader.program, block_name)
        if block_index != GL.GL_INVALID_INDEX:
            GL.glUniformBlockBinding(shader.program, block_index, binding_point)

    def on_resize(self):
        pass

    def render(self):
        # == per-frame auxiliary setup ==

        window_width, window_height = self.get_window_size()
        self.camera.aspect_ratio = window_width / window_height

        now = self.get_time()
        if self.last_update is None:
            self.last_update = now
        dt = now - self.last_update

        # == shader setup ==

        proj = self.camera.get_projection_matrix()
        view = self.camera.get_view_matrix()
        view_pos = self.camera.position

        proj_bytes = proj.tobytes()
        view_bytes = view.tobytes()
        extra_data = struct.pack('3ff', view_pos[0], view_pos[1], view_pos[2], now)

        ubo_data = proj_bytes + view_bytes + extra_data
        self.scene_ubo.update(ubo_data)

        # == gl clear ==

        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # == draw everything ==

        for entity in self.entities:
            entity.update(dt)

            entity.material.use()

            shader = entity.material.shader
            shader.set_vec3("u_LightPos", self.light_pos)
            shader.set_vec3("u_LightColor", self.light_color)

            model = entity.get_model_matrix()
            shader.set_mat4("u_Model", model)

            entity.mesh.draw()

        # == wrapoff ==

        self.last_update = now
