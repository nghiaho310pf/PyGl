import os
import threading
import queue
import ctypes
import numpy as np
import trimesh
import trimesh.transformations as tf
from PIL import Image
from OpenGL import GL

from entities.components.visuals.assets import (
    AssetsState, AssetStatus, Mesh, Texture, 
    ModelAsset, ModelNode, MaterialTemplate,
    ModelTask, MeshFileTask, MeshGeometryTask, 
    TextureFileTask, TextureImageTask,
    ModelResult, MeshResult, TextureResult
)
from entities.registry import Registry
from math_utils import vec3

def background_asset_worker(assets_state: AssetsState, task_queue: queue.Queue, result_queue: queue.Queue):
    while True:
        task = task_queue.get()

        match task:
            case ModelTask(asset_id, filepath):
                _process_model(assets_state, asset_id, filepath, result_queue, task_queue)
            case MeshFileTask(asset_id, filepath):
                _process_mesh_from_file(asset_id, filepath, result_queue)
            case MeshGeometryTask(asset_id, geom):
                _process_mesh_from_geom(asset_id, geom, result_queue)
            case TextureFileTask(asset_id, filepath, is_srgb):
                _process_texture_from_file(asset_id, filepath, is_srgb, result_queue)
            case TextureImageTask(asset_id, image, is_srgb):
                _process_texture_from_image(asset_id, image, is_srgb, result_queue)

        task_queue.task_done()


def _process_model(assets_state: AssetsState, asset_id: int, filepath: str, result_queue: queue.Queue, task_queue: queue.Queue):
    try:
        scene = trimesh.load_scene(filepath)
        nodes = []

        extension = os.path.splitext(filepath)[1].lower()
        is_gltf = extension in (".glb", ".gltf")
        if is_gltf:
            rotation = trimesh.transformations.rotation_matrix(np.radians(-180), [1, 0, 0])
            scene.apply_transform(rotation)

        for node_name in scene.graph.nodes_geometry:
            transform_matrix, geometry_name = scene.graph[node_name]
            geom = scene.geometry[geometry_name]

            pos = tf.translation_from_matrix(transform_matrix)
            euler_rad = tf.euler_from_matrix(transform_matrix)
            rot = np.degrees(euler_rad)  # type: ignore
            scale = tf.scale_from_matrix(transform_matrix)[0]

            virtual_mesh_id = AssetSystem.generate_id(assets_state)
            task_queue.put(MeshGeometryTask(virtual_mesh_id, geom))

            mat_template = MaterialTemplate()
            if hasattr(geom.visual, 'material'):
                mat = geom.visual.material

                if hasattr(mat, 'main_color'):
                    mat_template.albedo = (mat.main_color[:3] / 255.0) ** 2.2

                pil_image = None
                if hasattr(mat, 'image') and mat.image is not None:
                    pil_image = mat.image
                elif hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    pil_image = mat.baseColorTexture

                if pil_image is not None:
                    virtual_tex_id = AssetSystem.generate_id(assets_state)
                    task_queue.put(TextureImageTask(virtual_tex_id, pil_image, is_srgb=True))
                    mat_template.albedo_map_id = virtual_tex_id

            nodes.append(ModelNode(
                mesh_id=virtual_mesh_id,
                name=node_name,
                local_position=pos,
                local_rotation=rot,
                local_scale=vec3(scale, scale, scale),
                material_template=mat_template
            ))

        result_queue.put(ModelResult(asset_id=asset_id, nodes=nodes))
    except Exception as e:
        result_queue.put(ModelResult(asset_id=asset_id, error=e))


def _process_mesh_from_file(asset_id: int, filepath: str, result_queue: queue.Queue):
    try:
        geom = trimesh.load_mesh(filepath)

        extension = os.path.splitext(filepath)[1].lower()
        is_gltf = extension in (".glb", ".gltf")
        if is_gltf:
            rotation = trimesh.transformations.rotation_matrix(np.radians(-180), [1, 0, 0])
            geom.apply_transform(rotation)

        _process_mesh_from_geom(asset_id, geom, result_queue)
    except Exception as e:
        result_queue.put(MeshResult(asset_id=asset_id, error=e))


def _process_mesh_from_geom(asset_id: int, geom: trimesh.Trimesh, result_queue: queue.Queue):
    try:
        vertices = geom.vertices
        normals = geom.vertex_normals if (hasattr(geom, 'vertex_normals') and len(geom.vertex_normals) > 0) else np.zeros_like(vertices)

        if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:  # type: ignore
            uvs = geom.visual.uv.copy()  # type: ignore
        else:
            uvs = np.zeros((len(vertices), 2))

        interleaved = np.empty((len(vertices), 8), dtype=np.float32)
        interleaved[:, 0:3] = vertices
        interleaved[:, 3:6] = normals
        interleaved[:, 6:8] = uvs

        result_queue.put(MeshResult(
            asset_id=asset_id, 
            vertices=interleaved.ravel(), 
            indices=geom.faces.ravel().astype(np.uint32)
        ))
    except Exception as e:
        result_queue.put(MeshResult(asset_id=asset_id, error=e))


def _process_texture_from_file(asset_id: int, filepath: str, is_srgb: bool, result_queue: queue.Queue):
    _process_texture_from_image(asset_id, Image.open(filepath), is_srgb, result_queue)


def _process_texture_from_image(asset_id: int, img: Image.Image, is_srgb: bool, result_queue: queue.Queue):
    try:
        img_transposed = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if img_transposed.mode not in ("RGB", "RGBA"):
            img_transposed = img_transposed.convert("RGBA")

        format_ext = img_transposed.mode
        data = np.array(img_transposed)

        result_queue.put(TextureResult(asset_id=asset_id, data=data, format_info=format_ext, is_srgb=is_srgb))
    except Exception as e:
        result_queue.put(TextureResult(asset_id=asset_id, error=e))


class AssetSystem:
    @staticmethod
    def _ensure_worker(assets_state: AssetsState):
        if not assets_state.worker_started:
            thread = threading.Thread(
                target=background_asset_worker,
                args=(assets_state, assets_state.task_queue, assets_state.result_queue),
                daemon=True
            )
            thread.start()
            assets_state.worker_started = True
    
    @staticmethod
    def generate_id(assets_state: AssetsState) -> int:
        return next(assets_state.id_counter)

    @staticmethod
    def update(registry: Registry):
        r_assets = registry.get_singleton(AssetsState)
        if r_assets is None: return

        _, (assets_state, ) = r_assets
        max_uploads_per_frame = 3
        uploads = 0

        while uploads < max_uploads_per_frame:
            try:
                res = assets_state.result_queue.get_nowait()
            except queue.Empty:
                break

            uploads += 1

            match res:
                case ModelResult(asset_id, nodes, error):
                    model_obj = assets_state.models.get(asset_id)
                    if model_obj:
                        if error:
                            model_obj.status = AssetStatus.Failed
                            print(f"[AssetSystem] Error loading model: {error}")
                        elif nodes is not None:
                            model_obj.nodes = nodes
                            model_obj.status = AssetStatus.Ready
                            for node in nodes:
                                if node.mesh_id not in assets_state.meshes:
                                    assets_state.meshes[node.mesh_id] = Mesh(id=node.mesh_id, status=AssetStatus.Loading)
                                
                                tex_id = node.material_template.albedo_map_id
                                if tex_id is not None and tex_id not in assets_state.textures:
                                    assets_state.textures[tex_id] = Texture(id=tex_id, filepath="", status=AssetStatus.Loading, is_srgb=True)
                        else:
                            raise RuntimeError("AssetSystem: encountered illegal ModelResult")

                case MeshResult(asset_id, vertices, indices, error):
                    mesh_obj = assets_state.meshes.get(asset_id)
                    if mesh_obj is None:
                        mesh_obj = Mesh(id=asset_id, status=AssetStatus.Loading)
                        assets_state.meshes[asset_id] = mesh_obj

                    if error:
                        mesh_obj.status = AssetStatus.Failed
                        print(f"[AssetSystem] Error loading mesh: {error}")
                    elif vertices is not None:
                        AssetSystem._setup_gl_mesh(mesh_obj, vertices, indices)

                case TextureResult(asset_id, data, format_info, is_srgb, error):
                    tex_obj = assets_state.textures.get(asset_id)
                    if tex_obj is None:
                        tex_obj = Texture(id=asset_id, filepath="", status=AssetStatus.Loading)
                        assets_state.textures[asset_id] = tex_obj

                    if error:
                        tex_obj.status = AssetStatus.Failed
                        print(f"[AssetSystem] Error loading texture: {error}")
                    elif data is not None and format_info is not None:
                        tex_obj.is_srgb = is_srgb
                        AssetSystem._setup_gl_texture(tex_obj, data, format_info)
                    else:
                        raise RuntimeError("AssetSystem: encountered illegal TextureResult")

    @staticmethod
    def create_immediate_mesh(assets_state: AssetsState, vertices: np.ndarray, indices: np.ndarray) -> Mesh:
        asset_id = AssetSystem.generate_id(assets_state)
        mesh = Mesh(id=asset_id, status=AssetStatus.Loading)
        assets_state.meshes[asset_id] = mesh
        assets_state.result_queue.put(
            MeshResult(
                asset_id=asset_id, 
                vertices=vertices, 
                indices=indices, 
                error=None
            )
        )
        return mesh

    @staticmethod
    def request_model(assets_state: AssetsState, filepath: str) -> ModelAsset:
        AssetSystem._ensure_worker(assets_state)

        if filepath in assets_state.filepath_to_model:
            return assets_state.models[assets_state.filepath_to_model[filepath]]

        asset_id = AssetSystem.generate_id(assets_state)
        model = ModelAsset(id=asset_id, filepath=filepath, status=AssetStatus.Loading)

        assets_state.models[asset_id] = model
        assets_state.filepath_to_model[filepath] = asset_id
        assets_state.task_queue.put(ModelTask(asset_id, filepath))
        return model

    @staticmethod
    def request_mesh(assets_state: AssetsState, filepath_or_id: str | int) -> Mesh:
        AssetSystem._ensure_worker(assets_state)

        if isinstance(filepath_or_id, int):
            if filepath_or_id in assets_state.meshes:
                return assets_state.meshes[filepath_or_id]
            # pre-registration
            mesh = Mesh(id=filepath_or_id, status=AssetStatus.Loading)
            assets_state.meshes[filepath_or_id] = mesh
            return mesh

        filepath = filepath_or_id
        if filepath in assets_state.filepath_to_mesh:
            return assets_state.meshes[assets_state.filepath_to_mesh[filepath]]

        asset_id = AssetSystem.generate_id(assets_state)
        mesh = Mesh(id=asset_id, filepath=filepath, status=AssetStatus.Loading)
        assets_state.meshes[asset_id] = mesh
        assets_state.filepath_to_mesh[filepath] = asset_id
        assets_state.task_queue.put(MeshFileTask(asset_id, filepath))
        return mesh

    @staticmethod
    def request_texture(assets_state: AssetsState, filepath_or_id: str | int, is_srgb: bool = False) -> Texture:
        AssetSystem._ensure_worker(assets_state)
        
        if isinstance(filepath_or_id, int):
            if filepath_or_id in assets_state.textures:
                return assets_state.textures[filepath_or_id]
            tex = Texture(id=filepath_or_id, filepath="", status=AssetStatus.Loading, is_srgb=is_srgb)
            assets_state.textures[filepath_or_id] = tex
            return tex

        filepath = filepath_or_id
        if filepath in assets_state.filepath_to_texture:
            return assets_state.textures[assets_state.filepath_to_texture[filepath]]

        asset_id = AssetSystem.generate_id(assets_state)
        tex = Texture(id=asset_id, filepath=filepath, status=AssetStatus.Loading, is_srgb=is_srgb)
        assets_state.textures[asset_id] = tex
        assets_state.filepath_to_texture[filepath] = asset_id
        assets_state.task_queue.put(TextureFileTask(asset_id, filepath, is_srgb=is_srgb))
        return tex

    @staticmethod
    def _setup_gl_mesh(mesh: Mesh, vertices: np.ndarray, indices: np.ndarray | None):
        mesh.vao = GL.glGenVertexArrays(1)
        mesh.vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(mesh.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, mesh.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

        if indices is not None and len(indices) > 0:
            mesh.ebo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, mesh.ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
            mesh.indices_count = len(indices)
            mesh.has_indices = True
        else:
            mesh.has_indices = False
            mesh.indices_count = 0

        mesh.vertex_count = len(vertices) // 8
        stride = 8 * 4  # 8 floats, 4 bytes each

        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(24))

        GL.glBindVertexArray(0)
        mesh.status = AssetStatus.Ready

    @staticmethod
    def _setup_gl_texture(tex_obj: Texture, data: np.ndarray, format_info: str):
        tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        gl_format = GL.GL_RGB if format_info == "RGB" else GL.GL_RGBA
        internal_format = gl_format
        if tex_obj.is_srgb:
            internal_format = GL.GL_SRGB8 if gl_format == GL.GL_RGB else GL.GL_SRGB8_ALPHA8

        height, width = data.shape[:2]

        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, internal_format, width, height, 0,
            gl_format, GL.GL_UNSIGNED_BYTE, data
        )
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

        tex_obj.gl_id = tex_id
        tex_obj.width = width
        tex_obj.height = height
        tex_obj.status = AssetStatus.Ready
