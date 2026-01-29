from shader import Shader

PHONG_VS = """
#version 330 core
layout (location = 0) in vec3 a_Pos;
layout (location = 1) in vec3 a_Normal;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;

out vec3 v_Normal;
out vec3 v_FragPos;

void main() {
    v_FragPos = vec3(u_Model * vec4(a_Pos, 1.0));
    v_Normal = mat3(transpose(inverse(u_Model))) * a_Normal; 
    gl_Position = u_Projection * u_View * vec4(v_FragPos, 1.0);
}
"""

PHONG_FS = """
#version 330 core
out vec4 FragColor;

in vec3 v_Normal;
in vec3 v_FragPos;

uniform vec3 u_Color;
uniform vec3 u_LightPos;
uniform vec3 u_ViewPos;

void main() {
    vec3 norm = normalize(v_Normal);
    vec3 lightDir = normalize(u_LightPos - v_FragPos);
    float diff = max(dot(norm, lightDir), 0.0);

    vec3 result = (0.1 + diff) * u_Color;
    FragColor = vec4(result, 1.0);
}
"""


def make_shader():
    return Shader(PHONG_VS, PHONG_FS)
