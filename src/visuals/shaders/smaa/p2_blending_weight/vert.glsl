#version 450 core
#define SMAA_GLSL_4 1
#define SMAA_PRESET_ULTRA 1

#define SMAA_INCLUDE_VS 1
#define SMAA_INCLUDE_PS 0

layout(location = 0) in vec2 a_Position;
layout(location = 1) in vec2 a_TexCoord;

out vec2 v_TexCoord;
out vec2 v_PixCoord;
out vec4 v_Offset[3];

uniform vec4 SMAA_RT_METRICS;

// #inject

void main() {
    v_TexCoord = a_TexCoord;
    gl_Position = vec4(a_Position, 0.0, 1.0);
    SMAABlendingWeightCalculationVS(v_TexCoord, v_PixCoord, v_Offset);
}