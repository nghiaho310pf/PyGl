#version 450 core
out vec4 FragColor;

in vec2 v_UV;

uniform sampler2D u_Texture;
uniform sampler2D u_DepthTexture;
uniform sampler2D u_NormalTexture;
uniform bool u_Horizontal;
uniform float u_DepthSensitivity;
uniform float u_NormalThreshold;

uniform float u_Near;
uniform float u_Far;

const float weightCenter = 0.5;
const float weightEdge = 0.25;

float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * u_Near * u_Far) / (u_Far + u_Near - z * (u_Far - u_Near));
}

void main() {
    vec2 tex_offset = (1.0 / textureSize(u_Texture, 0));
    vec2 offset = u_Horizontal ? vec2(tex_offset.x, 0.0) : vec2(0.0, tex_offset.y);

    vec2 uvPos = v_UV + offset;
    vec2 uvNeg = v_UV - offset;

    float rawCenterDepth = texture(u_DepthTexture, v_UV).r;
    float centerDepth = linearizeDepth(rawCenterDepth);
    vec3 centerNormal = normalize(texture(u_NormalTexture, v_UV).rgb * 2.0 - 1.0);

    vec4 result = texture(u_Texture, v_UV) * weightCenter;
    float totalWeight = weightCenter;

    // positive offset
    float rawDepthPos = texture(u_DepthTexture, uvPos).r;
    float depthPos = linearizeDepth(rawDepthPos);
    vec3 normalPos = normalize(texture(u_NormalTexture, uvPos).rgb * 2.0 - 1.0);

    // reject neighbors with too different a normal
    float normalWeightPos = smoothstep(u_NormalThreshold, 1.0, dot(centerNormal, normalPos));

    float weightPos = weightEdge * max(0.0, 1.0 - abs(centerDepth - depthPos) * u_DepthSensitivity) * normalWeightPos;
    result += texture(u_Texture, uvPos) * weightPos;
    totalWeight += weightPos;

    // negative offset
    float rawDepthNeg = texture(u_DepthTexture, uvNeg).r;
    float depthNeg = linearizeDepth(rawDepthNeg);
    vec3 normalNeg = normalize(texture(u_NormalTexture, uvNeg).rgb * 2.0 - 1.0);

    float normalWeightNeg = smoothstep(u_NormalThreshold, 1.0, dot(centerNormal, normalNeg));

    float weightNeg = weightEdge * max(0.0, 1.0 - abs(centerDepth - depthNeg) * u_DepthSensitivity) * normalWeightNeg;
    result += texture(u_Texture, uvNeg) * weightNeg;
    totalWeight += weightNeg;

    FragColor = result / max(totalWeight, 0.0001);
}
