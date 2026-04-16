#version 450 core

in vec2 v_UV;

layout (location = 0) out vec4 out_PointShadows;
layout (location = 1) out vec4 out_DirShadows;

uniform sampler2D u_DepthTexture;
uniform sampler2D u_NormalTexture;
uniform sampler2D u_BlueNoiseTexture;
uniform mat4 u_InverseViewProjection;

layout (std140) uniform SceneData {
    mat4 u_Projection;
    mat4 u_View;
    vec3 u_ViewPos;
    float u_Time;
};

const int MAX_LIGHTS = 4;
const int MAX_VOGEL_SAMPLES = 64;

uniform int   u_PointPcfSamples;
uniform float u_InvPointPcfSamples;
uniform float u_InvSqrtPointPcfSamples;

uniform int   u_DirPcfSamples;
uniform float u_InvDirPcfSamples;
uniform float u_InvSqrtDirPcfSamples;

// VOGEL_DISK[i] = vec2(sqrt(i + 0.5), i * 2.4)
const vec2 VOGEL_DISK[MAX_VOGEL_SAMPLES] = vec2[](
    vec2(0.707106,   0.0), vec2(1.224744,   2.4), vec2(1.581138,   4.8), vec2(1.870828,   7.2),
    vec2(2.121320,   9.6), vec2(2.345207,  12.0), vec2(2.549509,  14.4), vec2(2.738612,  16.8),
    vec2(2.915475,  19.2), vec2(3.082207,  21.6), vec2(3.240370,  24.0), vec2(3.391164,  26.4),
    vec2(3.535533,  28.8), vec2(3.674234,  31.2), vec2(3.807886,  33.6), vec2(3.937003,  36.0),
    vec2(4.062019,  38.4), vec2(4.183300,  40.8), vec2(4.301162,  43.2), vec2(4.415880,  45.6),
    vec2(4.527692,  48.0), vec2(4.636809,  50.4), vec2(4.743416,  52.8), vec2(4.847679,  55.2),
    vec2(4.949747,  57.6), vec2(5.049752,  60.0), vec2(5.147815,  62.4), vec2(5.244044,  64.8),
    vec2(5.338539,  67.2), vec2(5.431390,  69.6), vec2(5.522680,  72.0), vec2(5.612486,  74.4),
    vec2(5.700877,  76.8), vec2(5.787918,  79.2), vec2(5.873670,  81.6), vec2(5.958187,  84.0),
    vec2(6.041522,  86.4), vec2(6.123724,  88.8), vec2(6.204836,  91.2), vec2(6.284902,  93.6),
    vec2(6.363961,  96.0), vec2(6.442049,  98.4), vec2(6.519202, 100.8), vec2(6.595452, 103.2),
    vec2(6.670832, 105.6), vec2(6.745368, 108.0), vec2(6.819090, 110.4), vec2(6.892024, 112.8),
    vec2(6.964194, 115.2), vec2(7.035623, 117.6), vec2(7.106335, 120.0), vec2(7.176350, 122.4),
    vec2(7.245688, 124.8), vec2(7.314369, 127.2), vec2(7.382411, 129.6), vec2(7.449832, 132.0),
    vec2(7.516648, 134.4), vec2(7.582875, 136.8), vec2(7.648529, 139.2), vec2(7.713624, 141.6),
    vec2(7.778174, 144.0), vec2(7.842193, 146.4), vec2(7.905694, 148.8), vec2(7.968688, 151.2)
);

uniform vec3 u_LightPos[MAX_LIGHTS];
uniform int u_NumLights;
uniform int u_PointLightCastsShadow[MAX_LIGHTS];
uniform samplerCube u_ShadowMap[MAX_LIGHTS];
uniform float u_FarPlane[MAX_LIGHTS];

uniform vec3 u_DirLightDirection[MAX_LIGHTS];
uniform int u_NumDirLights;
uniform int u_DirLightCastsShadow[MAX_LIGHTS];
uniform sampler2D u_DirShadowMap[MAX_LIGHTS];
uniform mat4 u_DirLightSpaceMatrix[MAX_LIGHTS];

const float PI2 = 6.28318530718;

float calculatePointShadow(vec3 fragPos, vec3 lightPos, samplerCube shadowMap, float farPlane, float bias, float randomRotation) {
    vec3 fragToLight = fragPos - lightPos;
    float currentDepth = length(fragToLight) / farPlane;

    vec3 lightDir = normalize(fragToLight);
    vec3 up = abs(lightDir.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(up, lightDir));
    up = cross(lightDir, right);

    float shadow = 0.0;
    for (int i = 0; i < u_PointPcfSamples; ++i) {
        float r = VOGEL_DISK[i].x * u_InvSqrtPointPcfSamples;
        float theta = VOGEL_DISK[i].y + randomRotation;
        vec2 offset2D = vec2(cos(theta), sin(theta)) * r;

        vec3 sampleDir = lightDir + (right * offset2D.x + up * offset2D.y) * 0.002;

        float closestDepth = texture(shadowMap, sampleDir).r;
        if (currentDepth - bias > closestDepth) shadow += 1.0;
    }

    return pow(shadow * u_InvPointPcfSamples, 2.4);
}

float calculateDirectionalShadow(vec4 fragPosLightSpace, sampler2D shadowMap, float bias, float randomRotation) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    if (projCoords.z > 1.0) return 0.0;

    float currentDepth = projCoords.z;
    float shadow = 0.0;
    for (int i = 0; i < u_DirPcfSamples; ++i) {
        float r = VOGEL_DISK[i].x * u_InvSqrtDirPcfSamples;
        float theta = VOGEL_DISK[i].y + randomRotation;
        vec2 offset = vec2(cos(theta), sin(theta)) * r * 0.001;

        float closestDepth = texture(shadowMap, projCoords.xy + offset).r;
        if (currentDepth - bias > closestDepth) shadow += 1.0;
    }

    return pow(shadow * u_InvDirPcfSamples, 2.4);
}

void main() {
    float depth = texture(u_DepthTexture, v_UV).r;
    if (depth == 1.0) discard;

    vec3 N = texture(u_NormalTexture, v_UV).xyz;

    vec4 ndc = vec4(v_UV * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 worldPosProj = u_InverseViewProjection * ndc;
    vec3 v_WorldPos = worldPosProj.xyz / worldPosProj.w;

    vec2 noiseUV = (gl_FragCoord.xy / 64.0) + fract(u_Time * vec2(0.13, 0.17));
    float globalNoise = texture(u_BlueNoiseTexture, noiseUV).r;
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
                    biasedWorldPos, u_LightPos[i], u_ShadowMap[i],
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
