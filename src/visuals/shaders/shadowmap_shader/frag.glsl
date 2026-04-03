#version 450 core
in vec3 v_FragPos;

uniform vec3 u_LightPos;
uniform float u_FarPlane;

void main() {
    float lightDistance = length(v_FragPos - u_LightPos);
    lightDistance = lightDistance / u_FarPlane;
    gl_FragDepth = lightDistance;
}