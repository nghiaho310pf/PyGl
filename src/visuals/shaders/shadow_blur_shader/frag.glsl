#version 450 core
out vec4 FragColor;

in vec2 v_UV;

uniform sampler2D u_Texture;
uniform sampler2D u_DepthTexture;
uniform sampler2D u_NormalTexture;

uniform vec2 u_TexelOffset;
uniform vec2 u_DepthParams;

uniform float u_DepthSensitivity;
uniform float u_NormalThreshold;

const float weightCenter = 0.5;
const float weightEdge = 0.25;

float linearizeDepth(float rawDepth) {
    float z = rawDepth * 2.0 - 1.0;
    return 1.0 / (u_DepthParams.x - z * u_DepthParams.y);
}

void main() {
    vec2 uvPos = v_UV + u_TexelOffset;
    vec2 uvNeg = v_UV - u_TexelOffset;

    float centerDepth = linearizeDepth(texture(u_DepthTexture, v_UV).r);
    vec3 centerNormal = texture(u_NormalTexture, v_UV).rgb * 2.0 - 1.0;
    vec4 result = texture(u_Texture, v_UV) * weightCenter;
    float totalWeight = weightCenter;

    // positive offset
    float depthPos = linearizeDepth(texture(u_DepthTexture, uvPos).r);
    vec3 normalPos = texture(u_NormalTexture, uvPos).rgb * 2.0 - 1.0;

    // reject neighbors with too different a normal
    float normalWeightPos = smoothstep(u_NormalThreshold, 1.0, dot(centerNormal, normalPos));

    float weightPos = weightEdge * max(0.0, 1.0 - abs(centerDepth - depthPos) * u_DepthSensitivity) * normalWeightPos;
    result += texture(u_Texture, uvPos) * weightPos;
    totalWeight += weightPos;

    // negative offset
    float depthNeg = linearizeDepth(texture(u_DepthTexture, uvNeg).r);
    vec3 normalNeg = texture(u_NormalTexture, uvNeg).rgb * 2.0 - 1.0;

    float normalWeightNeg = smoothstep(u_NormalThreshold, 1.0, dot(centerNormal, normalNeg));

    float weightNeg = weightEdge * max(0.0, 1.0 - abs(centerDepth - depthNeg) * u_DepthSensitivity) * normalWeightNeg;
    result += texture(u_Texture, uvNeg) * weightNeg;
    totalWeight += weightNeg;

    FragColor = result / max(totalWeight, 0.0001);
}
