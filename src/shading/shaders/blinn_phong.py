from shading.shader import Shader
from shading.src_utils import read_source_file


def make_shader():
    return Shader(read_source_file("blinn_phong/vert.glsl"), read_source_file("blinn_phong/frag.glsl"))
