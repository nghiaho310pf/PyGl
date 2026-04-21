#version 450 core
layout (location = 0) out uint FragColor;

layout (std140) uniform SceneData {
    mat4 u_Projection;
    mat4 u_View;
    vec3 u_ViewPos;
    float u_Time;
};

uniform uint u_EntityID;

void main() {
    FragColor = u_EntityID;
}
