#version 450 core
#define SMAA_GLSL_4 1
#define SMAA_PRESET_ULTRA 1

#define SMAA_INCLUDE_VS 0
#define SMAA_INCLUDE_PS 1

in vec2 v_TexCoord;
in vec2 v_PixCoord;
in vec4 v_Offset[3];

out vec4 out_Weights;

uniform sampler2D u_EdgeTex;
uniform sampler2D u_AreaTex;
uniform sampler2D u_SearchTex;
uniform vec4 SMAA_RT_METRICS;

// #inject

void main() {
    // The final vec4 is subsample indices. vec4(0.0) is standard for SMAA 1x
    out_Weights = SMAABlendingWeightCalculationPS(v_TexCoord, v_PixCoord, v_Offset, u_EdgeTex, u_AreaTex, u_SearchTex, vec4(0.0));
}