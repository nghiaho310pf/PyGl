#version 450 core
layout (location = 0) in vec3 a_Pos;

layout (std140) uniform SceneData {
    mat4 u_Projection;
    mat4 u_View;
    vec3 u_ViewPos;
    float u_Time;
};

uniform mat4 u_Model;

void main() {
    gl_Position = u_Projection * u_View * vec4(vec3(u_Model * vec4(a_Pos, 1.0)), 1.0);
}