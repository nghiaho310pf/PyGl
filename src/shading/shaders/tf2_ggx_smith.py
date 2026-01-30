from shading.shader import Shader
from shading.src_utils import read_source_file


def make_shader():
    return Shader(read_source_file("tf2_ggx_smith/vert.glsl"), read_source_file("tf2_ggx_smith/frag.glsl"))
