#version 450 core
out vec4 FragColor;

layout (std140) uniform SceneData {
    mat4 u_Projection;
    mat4 u_View;
    vec3 u_ViewPos;
    float u_Time;
};

uniform vec3 u_EntityColor;

void main() {
    FragColor = vec4(u_EntityColor, 1.0);
}
