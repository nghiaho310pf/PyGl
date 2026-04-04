#version 450 core
#define SMAA_GLSL_4 1
#define SMAA_PRESET_ULTRA 1

#define SMAA_INCLUDE_VS 0
#define SMAA_INCLUDE_PS 1

in vec2 v_TexCoord;
in vec4 v_Offset[3];

out vec2 out_Edges;

uniform sampler2D u_ColorTex;
uniform vec4 SMAA_RT_METRICS;

// #inject

void main() {
    out_Edges = SMAAColorEdgeDetectionPS(v_TexCoord, v_Offset, u_ColorTex);
}