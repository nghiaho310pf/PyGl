from pathlib import Path

from visuals.shader import Shader
from visuals.src_utils import read_source_file


def make_shaders():
    p = Path(__file__).parent.absolute()

    poly = read_source_file(p / "SMAA.hlsl")

    return (
        Shader(
            read_source_file(p / "p1_edge_detection/vert.glsl").replace("// #inject", poly),
            read_source_file(p / "p1_edge_detection/frag.glsl").replace("// #inject", poly)
        ),
        Shader(
            read_source_file(p / "p2_blending_weight/vert.glsl").replace("// #inject", poly),
            read_source_file(p / "p2_blending_weight/frag.glsl").replace("// #inject", poly)
        ),
        Shader(
            read_source_file(p / "p3_neighborhood_blending/vert.glsl").replace("// #inject", poly),
            read_source_file(p / "p3_neighborhood_blending/frag.glsl").replace("// #inject", poly)
        )
    )
