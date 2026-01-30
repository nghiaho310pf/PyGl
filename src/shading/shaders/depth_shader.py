from shading.shader import Shader

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 a_Pos;

uniform mat4 u_LightSpaceMatrix;
uniform mat4 u_Model;

void main() {
    gl_Position = u_LightSpaceMatrix * u_Model * vec4(a_Pos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
void main() {}
"""

def make_shader():
    return Shader(VERTEX_SHADER, FRAGMENT_SHADER)
