#version 450 core
layout (location = 0) in vec3 a_Pos;

uniform mat4 u_Projection;
uniform mat4 u_View;
uniform mat4 u_Model;

out vec3 v_FragPos;

void main() {
    v_FragPos = vec3(u_Model * vec4(a_Pos, 1.0));
    gl_Position = u_Projection * u_View * vec4(v_FragPos, 1.0);
}