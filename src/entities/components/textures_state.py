from dataclasses import dataclass, field
from queue import Queue
from enum import Enum, auto
import numpy.typing as npt


class TextureStatus(Enum):
    Unloaded = auto()
    Loading = auto()
    Ready = auto()
    Failed = auto()


@dataclass(slots=True, eq=False)
class Texture:
    filepath: str

    status: TextureStatus = TextureStatus.Unloaded
    gl_id: int | None = None
    width: int = 0
    height: int = 0


@dataclass(slots=True, eq=False)
class TexturesState:
    textures: dict[str, Texture] = field(default_factory=dict)
    upload_queue: Queue = field(default_factory=Queue)
