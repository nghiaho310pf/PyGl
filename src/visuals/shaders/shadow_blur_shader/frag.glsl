#version 450 core
out vec4 FragColor;

in vec2 v_UV;

uniform sampler2D u_Texture;
uniform sampler2D u_DepthTexture;
uniform sampler2D u_NormalTexture;
uniform bool u_Horizontal;
uniform float u_BlurScale;
uniform float u_DepthSensitivity;

const float weightCenter = 0.5;
const float weightEdge = 0.25;

void main() {
    vec2 tex_offset = (1.0 / textureSize(u_Texture, 0)) * u_BlurScale;
    vec2 offset = u_Horizontal ? vec2(tex_offset.x, 0.0) : vec2(0.0, tex_offset.y);

    vec2 uvPos = v_UV + offset;
    vec2 uvNeg = v_UV - offset;

    float centerDepth = texture(u_DepthTexture, v_UV).r;
    vec3 centerNormal = normalize(texture(u_NormalTexture, v_UV).rgb);

    vec4 result = texture(u_Texture, v_UV) * weightCenter;
    float totalWeight = weightCenter;

    // positive offset
    float depthPos = texture(u_DepthTexture, uvPos).r;
    vec3 normalPos = normalize(texture(u_NormalTexture, uvPos).rgb);

    // create a strict pow(8) falloff so curves blur, but corners don't
    float normalWeightPos = pow(max(0.0, dot(centerNormal, normalPos)), 8.0);

    float weightPos = weightEdge * max(0.0, 1.0 - abs(centerDepth - depthPos) * u_DepthSensitivity) * normalWeightPos;
    result += texture(u_Texture, uvPos) * weightPos;
    totalWeight += weightPos;

    // negative offset
    float depthNeg = texture(u_DepthTexture, uvNeg).r;
    vec3 normalNeg = normalize(texture(u_NormalTexture, uvNeg).rgb);

    float normalWeightNeg = pow(max(0.0, dot(centerNormal, normalNeg)), 8.0);

    float weightNeg = weightEdge * max(0.0, 1.0 - abs(centerDepth - depthNeg) * u_DepthSensitivity) * normalWeightNeg;
    result += texture(u_Texture, uvNeg) * weightNeg;
    totalWeight += weightNeg;

    FragColor = result / max(totalWeight, 0.0001);
}
