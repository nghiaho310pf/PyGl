from shading.shader import Shader

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 a_Pos;

uniform mat4 u_Projection;
uniform mat4 u_View;
uniform mat4 u_Model;

out vec3 v_FragPos;

void main() {
    v_FragPos = vec3(u_Model * vec4(a_Pos, 1.0));
    gl_Position = u_Projection * u_View * vec4(v_FragPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 v_FragPos;

uniform vec3 u_LightPos;
uniform float u_FarPlane;

void main() {
    float lightDistance = length(v_FragPos - u_LightPos);
    lightDistance = lightDistance / u_FarPlane;
    gl_FragDepth = lightDistance;
}
"""

def make_shader():
    return Shader(VERTEX_SHADER, FRAGMENT_SHADER)
