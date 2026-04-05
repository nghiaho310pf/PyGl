#version 450 core
layout (location = 0) out vec3 f_Normal;

in vec3 v_Normal;

void main() {
    f_Normal = normalize(v_Normal);
}
