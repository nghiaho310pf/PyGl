from enum import Enum, auto
import threading
import queue
import ctypes
import numpy as np
import trimesh
import trimesh.transformations as tf
from PIL import Image
from OpenGL import GL

from entities.components.visuals.assets import AssetsState, AssetStatus, Mesh, Texture, ModelAsset, ModelNode, MaterialTemplate
from entities.registry import Registry
from math_utils import vec3


class TaskType(Enum):
    Mesh = auto()
    Texture = auto()
    Model = auto()


def background_asset_worker(assets_state: AssetsState, task_queue: queue.Queue, result_queue: queue.Queue):
    while True:
        task_type, asset_id, filepath, payload = task_queue.get()

        if task_type == TaskType.Mesh:
            _process_mesh(asset_id, filepath, payload, result_queue)
        elif task_type == TaskType.Texture:
            _process_texture(asset_id, filepath, payload, result_queue)
        elif task_type == TaskType.Model:
            _process_model(assets_state, asset_id, filepath, result_queue, task_queue)

        task_queue.task_done()


def _process_model(assets_state: AssetsState, asset_id: int, filepath: str, result_queue: queue.Queue, task_queue: queue.Queue):
    try:
        is_gltf = filepath.lower().endswith(('.glb', '.gltf'))

        scene = trimesh.load(filepath, force='scene')
        nodes = []

        for node_name in scene.graph.nodes_geometry:  # type: ignore
            transform_matrix, geometry_name = scene.graph[node_name]  # type: ignore
            geom = scene.geometry[geometry_name]  # type: ignore

            pos = tf.translation_from_matrix(transform_matrix)
            euler_rad = tf.euler_from_matrix(transform_matrix)
            rot = np.degrees(euler_rad)  # type: ignore
            scale = tf.scale_from_matrix(transform_matrix)[0]

            virtual_mesh_id = AssetSystem.generate_id(assets_state)
            task_queue.put((TaskType.Mesh, virtual_mesh_id, None, (geom, is_gltf)))

            mat_template = MaterialTemplate()
            if hasattr(geom.visual, 'material'):
                mat = geom.visual.material

                if hasattr(mat, 'main_color'):
                    mat_template.albedo = mat.main_color[:3] / 255.0

                pil_image = None
                if hasattr(mat, 'image') and mat.image is not None:
                    pil_image = mat.image
                elif hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    pil_image = mat.baseColorTexture

                if pil_image is not None:
                    virtual_tex_id = AssetSystem.generate_id(assets_state)
                    task_queue.put((TaskType.Texture, virtual_tex_id, None, pil_image))
                    mat_template.albedo_map_id = virtual_tex_id

            nodes.append(ModelNode(
                mesh_id=virtual_mesh_id,
                name=node_name,
                local_position=pos,
                local_rotation=rot,
                local_scale=vec3(scale, scale, scale),
                material_template=mat_template
            ))

        result_queue.put((TaskType.Model, asset_id, (nodes, None)))
    except Exception as e:
        result_queue.put((TaskType.Model, asset_id, (None, e)))


def _process_mesh(asset_id: int, filepath: str | None, payload, result_queue: queue.Queue):
    try:
        if isinstance(payload, tuple):
            geom, flip_uvs = payload
        else:
            geom = payload
            flip_uvs = filepath is not None and filepath.lower().endswith(('.glb', '.gltf'))

        if geom is None and filepath is not None:  
            geom = trimesh.load(filepath)

        rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        geom.apply_transform(rotation)

        vertices = geom.vertices  # type: ignore
        if hasattr(geom, 'vertex_normals') and len(geom.vertex_normals) > 0:  # type: ignore
            normals = geom.vertex_normals  # type: ignore
        else:
            normals = np.zeros_like(vertices)
        if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:  # type: ignore
            uvs = geom.visual.uv.copy()  # type: ignore
            if flip_uvs:
                uvs[:, 1] = 1.0 - uvs[:, 1]
        else:
            uvs = np.zeros((len(vertices), 2))

        interleaved = np.empty((len(vertices), 8), dtype=np.float32)
        interleaved[:, 0:3] = vertices
        interleaved[:, 3:6] = normals
        interleaved[:, 6:8] = uvs

        interleaved = interleaved.ravel()
        indices = geom.faces.ravel().astype(np.uint32)  # type: ignore

        result_queue.put((TaskType.Mesh, asset_id, (interleaved, indices, None)))
    except Exception as e:
        result_queue.put((TaskType.Mesh, asset_id, (None, None, e)))


def _process_texture(asset_id: int, filepath: str | None, payload, result_queue: queue.Queue):
    try:
        img = payload if isinstance(payload, Image.Image) else Image.open(filepath)  # type: ignore

        img_transposed = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if img_transposed.mode not in ("RGB", "RGBA"):
            img_transposed = img_transposed.convert("RGBA")

        format_ext = img_transposed.mode
        data = np.array(img_transposed)

        if not isinstance(payload, Image.Image):
            img.close()

        result_queue.put((TaskType.Texture, asset_id, (data, format_ext, None)))
    except Exception as e:
        result_queue.put((TaskType.Texture, asset_id, (None, None, e)))


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
                task_type, asset_id, result = assets_state.result_queue.get_nowait()
            except queue.Empty:
                break

            uploads += 1

            if task_type == TaskType.Model:
                nodes, error = result
                model_obj = assets_state.models.get(asset_id)
                if model_obj:
                    if error:
                        model_obj.status = AssetStatus.Failed
                        print(f"[AssetSystem] Error loading model: {error}")
                    else:
                        model_obj.nodes = nodes
                        model_obj.status = AssetStatus.Ready

                        for node in nodes:
                            if node.mesh_id not in assets_state.meshes:
                                assets_state.meshes[node.mesh_id] = Mesh(id=node.mesh_id, status=AssetStatus.Loading)
                            
                            tex_id = node.material_template.albedo_map_id
                            if tex_id is not None and tex_id not in assets_state.textures:
                                assets_state.textures[tex_id] = Texture(id=tex_id, filepath="", status=AssetStatus.Loading)

            elif task_type == TaskType.Mesh:
                vertices, indices, error = result
                mesh_obj = assets_state.meshes.get(asset_id)

                if mesh_obj is None:
                    # sub-asset finished loading before its parent did
                    mesh_obj = Mesh(id=asset_id, status=AssetStatus.Loading)
                    assets_state.meshes[asset_id] = mesh_obj

                if error:
                    mesh_obj.status = AssetStatus.Failed
                    print(f"[AssetSystem] Error loading mesh: {error}")
                else:
                    AssetSystem._setup_gl_mesh(mesh_obj, vertices, indices)

            elif task_type == TaskType.Texture:
                data, format_info, error = result
                tex_obj = assets_state.textures.get(asset_id)

                if tex_obj is None:
                    tex_obj = Texture(id=asset_id, filepath="", status=AssetStatus.Loading)
                    assets_state.textures[asset_id] = tex_obj

                if error:
                    tex_obj.status = AssetStatus.Failed
                    print(f"[AssetSystem] Error loading texture: {error}")
                else:
                    AssetSystem._setup_gl_texture(tex_obj, data, format_info)

    @staticmethod
    def create_immediate_mesh(assets_state: AssetsState, vertices: np.ndarray, indices: np.ndarray) -> Mesh:
        asset_id = AssetSystem.generate_id(assets_state)
        mesh = Mesh(id=asset_id, status=AssetStatus.Loading)
        assets_state.meshes[asset_id] = mesh
        assets_state.result_queue.put((TaskType.Mesh, asset_id, (vertices, indices, None)))
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

        assets_state.task_queue.put((TaskType.Model, asset_id, filepath, None))
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
        
        assets_state.task_queue.put((TaskType.Mesh, asset_id, filepath, None))
        return mesh

    @staticmethod
    def request_texture(assets_state: AssetsState, filepath_or_id: str | int) -> Texture:
        AssetSystem._ensure_worker(assets_state)
        
        if isinstance(filepath_or_id, int):
            if filepath_or_id in assets_state.textures:
                return assets_state.textures[filepath_or_id]
            tex = Texture(id=filepath_or_id, filepath="", status=AssetStatus.Loading)
            assets_state.textures[filepath_or_id] = tex
            return tex

        filepath = filepath_or_id
        if filepath in assets_state.filepath_to_texture:
            return assets_state.textures[assets_state.filepath_to_texture[filepath]]

        asset_id = AssetSystem.generate_id(assets_state)
        tex = Texture(id=asset_id, filepath=filepath, status=AssetStatus.Loading)
        assets_state.textures[asset_id] = tex
        assets_state.filepath_to_texture[filepath] = asset_id

        assets_state.task_queue.put((TaskType.Texture, asset_id, filepath, None))
        return tex

    @staticmethod
    def _setup_gl_mesh(mesh: Mesh, vertices: np.ndarray, indices: np.ndarray | None):
        mesh.vao = GL.glGenVertexArrays(1)
        mesh.vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(mesh.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, mesh.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes,
                        vertices, GL.GL_STATIC_DRAW)

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
        height, width = data.shape[:2]

        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, gl_format, width, height, 0,
            gl_format, GL.GL_UNSIGNED_BYTE, data
        )
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

        tex_obj.gl_id = tex_id
        tex_obj.width = width
        tex_obj.height = height
        tex_obj.status = AssetStatus.Ready
