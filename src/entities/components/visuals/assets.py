from dataclasses import dataclass, field
from enum import Enum, auto
import queue
import itertools
from typing import Iterator, Union

import numpy as np
import numpy.typing as npt
import trimesh
from PIL import Image


class AssetStatus(Enum):
    Unloaded = auto()
    Loading = auto()
    Ready = auto()
    Failed = auto()


# == task types sent from main thread to worker thread ==


@dataclass(slots=True)
class ModelTask:
    asset_id: int
    filepath: str


@dataclass(slots=True)
class MeshFileTask:
    asset_id: int
    filepath: str


@dataclass(slots=True)
class MeshGeometryTask:
    asset_id: int
    geometry: trimesh.Trimesh


@dataclass(slots=True)
class TextureFileTask:
    asset_id: int
    filepath: str
    is_srgb: bool = False


@dataclass(slots=True)
class TextureImageTask:
    asset_id: int
    image: Image.Image
    is_srgb: bool = False


AssetTask = Union[ModelTask, MeshFileTask, MeshGeometryTask, TextureFileTask, TextureImageTask]


# == result types sent from worker thread to main thread ==


@dataclass(slots=True)
class ModelResult:
    asset_id: int
    nodes: list["ModelNode"] | None = None
    error: Exception | None = None


@dataclass(slots=True)
class MeshResult:
    asset_id: int
    vertices: np.ndarray | None = None
    indices: np.ndarray | None = None
    error: Exception | None = None


@dataclass(slots=True)
class TextureResult:
    asset_id: int
    data: np.ndarray | None = None
    format_info: str | None = None
    is_srgb: bool = False
    error: Exception | None = None


AssetResult = Union[ModelResult, MeshResult, TextureResult]


# == asset types ==


@dataclass(slots=True, eq=False)
class Mesh:
    id: int
    filepath: str | None = None
    status: AssetStatus = AssetStatus.Unloaded
    vao: int = 0
    vbo: int = 0
    ebo: int = 0
    vertex_count: int = 0
    indices_count: int = 0
    has_indices: bool = False


@dataclass(slots=True, eq=False)
class Texture:
    id: int
    filepath: str
    status: AssetStatus = AssetStatus.Unloaded
    gl_id: int | None = None
    width: int = 0
    height: int = 0
    is_srgb: bool = False


@dataclass(slots=True, eq=False)
class MaterialTemplate:
    albedo: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([0.4, 0.4, 0.4], dtype=np.float32))
    roughness: np.float32 = field(default_factory=lambda: np.float32(0.6))
    metallic: np.float32 = field(default_factory=lambda: np.float32(0.0))
    reflectance: np.float32 = field(default_factory=lambda: np.float32(0.0))
    translucency: np.float32 = field(default_factory=lambda: np.float32(0.0))
    ao: np.float32 = field(default_factory=lambda: np.float32(0.12))
    albedo_map_id: int | None = None
    normal_map_id: int | None = None
    roughness_map_id: int | None = None
    metallic_map_id: int | None = None


@dataclass(slots=True, eq=False)
class ModelNode:
    mesh_id: int
    name: str
    local_position: npt.NDArray[np.float32]
    local_rotation: npt.NDArray[np.float32]
    local_scale: npt.NDArray[np.float32]
    material_template: MaterialTemplate


@dataclass(slots=True, eq=False)
class ModelAsset:
    id: int
    filepath: str
    status: AssetStatus = AssetStatus.Unloaded
    nodes: list[ModelNode] = field(default_factory=list)


@dataclass(slots=True, eq=False)
class AssetsState:
    worker_started: bool = False

    id_counter: Iterator[int] = field(default_factory=lambda: itertools.count(1))

    meshes: dict[int, Mesh] = field(default_factory=dict)
    textures: dict[int, Texture] = field(default_factory=dict)
    models: dict[int, ModelAsset] = field(default_factory=dict)

    filepath_to_mesh: dict[str, int] = field(default_factory=dict)
    filepath_to_texture: dict[str, int] = field(default_factory=dict)
    filepath_to_model: dict[str, int] = field(default_factory=dict)

    task_queue: queue.Queue[AssetTask] = field(default_factory=queue.Queue)
    result_queue: queue.Queue[AssetResult] = field(default_factory=queue.Queue)
