#version 450 core
layout (location = 0) in vec3 a_Pos;
layout (location = 1) in vec3 a_Normal;

layout (std140) uniform SceneData {
    mat4 u_Projection;
    mat4 u_View;
    vec3 u_ViewPos;
    float u_Time;
};

uniform mat4 u_Model;

out vec3 v_Normal;
out vec3 v_FragPos;

void main() {
    v_FragPos = vec3(u_Model * vec4(a_Pos, 1.0));
    v_Normal = mat3(transpose(inverse(u_Model))) * a_Normal; 
    gl_Position = u_Projection * u_View * vec4(v_FragPos, 1.0);
}