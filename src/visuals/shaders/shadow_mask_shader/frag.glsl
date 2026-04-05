#version 450 core

in vec2 v_UV;

layout (location = 0) out vec4 out_PointShadows;
layout (location = 1) out vec4 out_DirShadows;

uniform sampler2D u_DepthTexture;
uniform sampler2D u_NormalTexture;
uniform mat4 u_InverseViewProjection;

layout (std140) uniform SceneData {
    mat4 u_Projection;
    mat4 u_View;
    vec3 u_ViewPos;
    float u_Time;
};

const int MAX_LIGHTS = 4;
uniform vec3 u_LightPos[MAX_LIGHTS];
uniform float u_LightRadius[MAX_LIGHTS];
uniform int u_NumLights;
uniform int u_PointLightCastsShadow[MAX_LIGHTS];
uniform samplerCube u_ShadowMap[MAX_LIGHTS];
uniform float u_FarPlane[MAX_LIGHTS];

uniform vec3 u_DirLightDirection[MAX_LIGHTS];
uniform int u_NumDirLights;
uniform int u_DirLightCastsShadow[MAX_LIGHTS];
uniform sampler2D u_DirShadowMap[MAX_LIGHTS];
uniform mat4 u_DirLightSpaceMatrix[MAX_LIGHTS];

const float PI = 3.14159265359;
const float PI2 = 6.28318530718;

float blueNoiseDither(vec2 pos) {
    float white = fract(sin(dot(pos, vec2(12.9898, 78.233))) * 43758.5453);
    float noise = fract(white + (gl_FragCoord.x + gl_FragCoord.y * 0.5) * 0.375);
    return noise;
}

float calculatePointShadow(vec3 fragPos, vec3 lightPos, float lightRadius, samplerCube shadowMap, float farPlane, float bias, float randomRotation) {
    vec3 fragToLight = fragPos - lightPos;
    float currentDepth = length(fragToLight) / farPlane;

    int searchSamples = 4;
    float avgBlockerDepth = 0.0;
    int blockers = 0;
    float searchWidth = lightRadius * 0.5;

    vec3 lightDir = normalize(fragToLight);
    vec3 up = abs(lightDir.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(up, lightDir));
    up = cross(lightDir, right);

    const float GOLDEN_ANGLE = 2.4;
    const mat2 vogelRot = mat2(cos(GOLDEN_ANGLE), sin(GOLDEN_ANGLE), -sin(GOLDEN_ANGLE), cos(GOLDEN_ANGLE));
    
    vec2 vogelDir = vec2(cos(randomRotation), sin(randomRotation));
    float invSqrtSearchSamples = 1.0 / sqrt(float(searchSamples));

    for (int i = 0; i < searchSamples; ++i) {
        float r = sqrt(float(i) + 0.5) * invSqrtSearchSamples;
        vec2 offset2D = vogelDir * r;
        vogelDir = vogelRot * vogelDir;

        vec3 sampleDir = lightDir + (right * offset2D.x + up * offset2D.y) * searchWidth;

        float sampleDepth = texture(shadowMap, sampleDir).r;
        if (sampleDepth < currentDepth - bias) {
            avgBlockerDepth += sampleDepth;
            blockers++;
        }
    }

    if (blockers < 1) return 0.0;
    avgBlockerDepth /= float(blockers);

    float penumbraRatio = (currentDepth - avgBlockerDepth) / pow(avgBlockerDepth, 0.7);
    float diskRadius = penumbraRatio * lightRadius;
    diskRadius = clamp(diskRadius, 0.005, 0.05);

    float shadow = 0.0;
    int pcfSamples = 8;
    float invPcfSamples = 1.0 / float(pcfSamples);
    float invSqrtPcfSamples = 1.0 / sqrt(float(pcfSamples));

    vogelDir = vec2(cos(randomRotation), sin(randomRotation));

    for (int i = 0; i < pcfSamples; ++i) {
        float r = sqrt(float(i) + 0.5) * invSqrtPcfSamples;
        vec2 offset2D = vogelDir * r;
        vogelDir = vogelRot * vogelDir;

        vec3 sampleDir = lightDir + (right * offset2D.x + up * offset2D.y) * diskRadius;

        float closestDepth = texture(shadowMap, sampleDir).r;
        if (currentDepth - bias > closestDepth) shadow += 1.0;
    }

    return shadow * invPcfSamples;
}

float calculateDirectionalShadow(vec4 fragPosLightSpace, sampler2D shadowMap, float bias, float randomRotation) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    if (projCoords.z > 1.0) return 0.0;

    float currentDepth = projCoords.z;

    int searchSamples = 8;
    float avgBlockerDepth = 0.0;
    int blockers = 0;

    float searchWidth = 0.005; 

    const float GOLDEN_ANGLE = 2.4;
    const mat2 vogelRot = mat2(cos(GOLDEN_ANGLE), sin(GOLDEN_ANGLE), -sin(GOLDEN_ANGLE), cos(GOLDEN_ANGLE));
    vec2 vogelDir = vec2(cos(randomRotation), sin(randomRotation));
    float invSqrtSearchSamples = 1.0 / sqrt(float(searchSamples));

    for (int i = 0; i < searchSamples; ++i) {
        float r = sqrt(float(i) + 0.5) * invSqrtSearchSamples;
        vec2 offset = vogelDir * r * searchWidth;
        vogelDir = vogelRot * vogelDir;

        float sampleDepth = texture(shadowMap, projCoords.xy + offset).r;
        if (sampleDepth < currentDepth - bias) {
            avgBlockerDepth += sampleDepth;
            blockers++;
        }
    }

    if (blockers < 2) return 0.0;
    avgBlockerDepth /= float(blockers);

    float penumbra = ((currentDepth - avgBlockerDepth) / max(avgBlockerDepth, 0.025)) * 0.025;
    penumbra = clamp(penumbra, 0.0005, 0.01);

    float shadow = 0.0;
    int pcfSamples = 16;
    float invPcfSamples = 1.0 / float(pcfSamples);
    float invSqrtPcfSamples = 1.0 / sqrt(float(pcfSamples));
    
    vogelDir = vec2(cos(randomRotation), sin(randomRotation));

    for (int i = 0; i < pcfSamples; ++i) {
        float r = sqrt(float(i) + 0.5) * invSqrtPcfSamples;
        vec2 offset = vogelDir * r * penumbra;
        vogelDir = vogelRot * vogelDir;

        float closestDepth = texture(shadowMap, projCoords.xy + offset).r;
        if (currentDepth - bias > closestDepth) {
            shadow += 1.0;
        }
    }
    
    return shadow * invPcfSamples;
}

void main() {
    float depth = texture(u_DepthTexture, v_UV).r;
    if (depth == 1.0) discard;

    vec3 N = texture(u_NormalTexture, v_UV).xyz;
    
    vec4 ndc = vec4(v_UV * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 worldPosProj = u_InverseViewProjection * ndc;
    vec3 v_WorldPos = worldPosProj.xyz / worldPosProj.w;

    float globalNoise = blueNoiseDither(gl_FragCoord.xy * fract(u_Time * 2.5));
    float randomRotation = globalNoise * PI2;

    vec4 pointShadows = vec4(0.0);
    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= u_NumLights) break;
        if (u_PointLightCastsShadow[i] == 1) {
            float distance = length(u_LightPos[i] - v_WorldPos);
            if (distance <= u_FarPlane[i]) {
                vec3 L = normalize(u_LightPos[i] - v_WorldPos);
                float normalOffsetScale = max(0.02 * (1.0 - dot(N, L)), 0.002);
                vec3 biasedWorldPos = v_WorldPos + N * normalOffsetScale;
                float shadow = calculatePointShadow(
                    biasedWorldPos, u_LightPos[i], u_LightRadius[i], u_ShadowMap[i],
                    u_FarPlane[i], 0.00025, randomRotation
                );
                pointShadows[i] = smoothstep(0.01, 0.98, shadow);
            }
        }
    }

    vec4 dirShadows = vec4(0.0);
    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= u_NumDirLights) break;
        if (u_DirLightCastsShadow[i] == 1) {
            vec3 L = normalize(-u_DirLightDirection[i]);
            float normalOffsetScale = max(0.02 * (1.0 - dot(N, L)), 0.002);
            vec3 biasedWorldPos = v_WorldPos + N * normalOffsetScale;
            vec4 fragPosLightSpace = u_DirLightSpaceMatrix[i] * vec4(biasedWorldPos, 1.0);
            dirShadows[i] = calculateDirectionalShadow(
                fragPosLightSpace, u_DirShadowMap[i], 0.00025, randomRotation
            );
        }
    }

    out_PointShadows = pointShadows;
    out_DirShadows = dirShadows;
}
