#version 450 core
out vec4 FragColor;

in vec3 v_Normal;
in vec3 v_FragPos;

layout (std140) uniform SceneData {
    mat4 u_Projection;
    mat4 u_View;
    vec3 u_ViewPos;
    float u_Time;
};

uniform mat4 u_Model;
uniform vec3 u_Albedo;

float filmGrain(vec2 coords) {
    return fract(sin(dot(coords.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    FragColor = vec4(
        u_Albedo + (filmGrain(gl_FragCoord.xy + fract(u_Time)) - 0.5) * 0.002,
        1.0
    );
}