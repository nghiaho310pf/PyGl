#version 450 core
#define SMAA_GLSL_4 1
#define SMAA_PRESET_ULTRA 1

#define SMAA_INCLUDE_VS 0
#define SMAA_INCLUDE_PS 1

in vec2 v_TexCoord;
in vec4 v_Offset;

out vec4 out_FragColor;

uniform sampler2D u_ColorTex;
uniform sampler2D u_BlendTex;
uniform vec4 SMAA_RT_METRICS;

// #inject

void main() {
    out_FragColor = SMAANeighborhoodBlendingPS(v_TexCoord, v_Offset, u_ColorTex, u_BlendTex);
}