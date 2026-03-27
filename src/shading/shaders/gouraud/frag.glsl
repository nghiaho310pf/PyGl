#version 450 core
out vec4 FragColor;

in vec3 v_Color;

layout (std140) uniform SceneData {
    mat4 u_Projection;
    mat4 u_View;
    vec3 u_ViewPos;
    float u_Time;
};

float filmGrain(vec2 coords) {
    return fract(sin(dot(coords.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    FragColor = vec4(
        v_Color + (filmGrain(gl_FragCoord.xy + fract(u_Time)) - 0.5) * 0.002,
        1.0
    );
}