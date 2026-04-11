from pathlib import Path

from visuals.shader import Shader
from visuals.src_utils import read_source_file


def make_shader():
    p = Path(__file__).parent.absolute()
    return Shader(
        read_source_file(p / "id_shader/vert.glsl"),
        read_source_file(p / "id_shader/frag.glsl")
    )
