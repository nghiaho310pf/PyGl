from pathlib import Path

from shading.shader import Shader
from shading.src_utils import read_source_file


def make_shader():
    p = Path(__file__).parent.absolute()
    return Shader(
        read_source_file(p / "tf2_ggx_smith/vert.glsl"),
        read_source_file(p / "tf2_ggx_smith/frag.glsl")
    )
