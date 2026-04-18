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
const int MAX_VOGEL_SAMPLES = 128;

uniform int   u_PointPcfSamples;
uniform float u_InvPointPcfSamples;
uniform float u_InvSqrtPointPcfSamples;

uniform int   u_DirPcfSamples;
uniform float u_InvDirPcfSamples;
uniform float u_InvSqrtDirPcfSamples;

// VOGEL_DISK[i] = vec2(sqrt(i + 0.5), i * 2.4)
const vec2 VOGEL_DISK[MAX_VOGEL_SAMPLES] = vec2[](
    vec2( 0.707107,   0.0), vec2( 1.224745,   2.4), vec2( 1.581139,   4.8), vec2( 1.870829,   7.2),
    vec2( 2.121320,   9.6), vec2( 2.345208,  12.0), vec2( 2.549510,  14.4), vec2( 2.738613,  16.8),
    vec2( 2.915476,  19.2), vec2( 3.082207,  21.6), vec2( 3.240370,  24.0), vec2( 3.391165,  26.4),
    vec2( 3.535534,  28.8), vec2( 3.674235,  31.2), vec2( 3.807887,  33.6), vec2( 3.937004,  36.0),
    vec2( 4.062019,  38.4), vec2( 4.183300,  40.8), vec2( 4.301163,  43.2), vec2( 4.415880,  45.6),
    vec2( 4.527693,  48.0), vec2( 4.636809,  50.4), vec2( 4.743416,  52.8), vec2( 4.847680,  55.2),
    vec2( 4.949747,  57.6), vec2( 5.049752,  60.0), vec2( 5.147815,  62.4), vec2( 5.244044,  64.8),
    vec2( 5.338539,  67.2), vec2( 5.431390,  69.6), vec2( 5.522681,  72.0), vec2( 5.612486,  74.4),
    vec2( 5.700877,  76.8), vec2( 5.787918,  79.2), vec2( 5.873670,  81.6), vec2( 5.958188,  84.0),
    vec2( 6.041523,  86.4), vec2( 6.123724,  88.8), vec2( 6.204837,  91.2), vec2( 6.284903,  93.6),
    vec2( 6.363961,  96.0), vec2( 6.442049,  98.4), vec2( 6.519202, 100.8), vec2( 6.595453, 103.2),
    vec2( 6.670832, 105.6), vec2( 6.745369, 108.0), vec2( 6.819091, 110.4), vec2( 6.892024, 112.8),
    vec2( 6.964194, 115.2), vec2( 7.035624, 117.6), vec2( 7.106335, 120.0), vec2( 7.176350, 122.4),
    vec2( 7.245688, 124.8), vec2( 7.314369, 127.2), vec2( 7.382412, 129.6), vec2( 7.449832, 132.0),
    vec2( 7.516648, 134.4), vec2( 7.582875, 136.8), vec2( 7.648529, 139.2), vec2( 7.713624, 141.6),
    vec2( 7.778175, 144.0), vec2( 7.842194, 146.4), vec2( 7.905694, 148.8), vec2( 7.968689, 151.2),
    vec2( 8.031189, 153.6), vec2( 8.093207, 156.0), vec2( 8.154753, 158.4), vec2( 8.215838, 160.8),
    vec2( 8.276473, 163.2), vec2( 8.336666, 165.6), vec2( 8.396428, 168.0), vec2( 8.455767, 170.4),
    vec2( 8.514693, 172.8), vec2( 8.573214, 175.2), vec2( 8.631338, 177.6), vec2( 8.689074, 180.0),
    vec2( 8.746428, 182.4), vec2( 8.803408, 184.8), vec2( 8.860023, 187.2), vec2( 8.916277, 189.6),
    vec2( 8.972179, 192.0), vec2( 9.027735, 194.4), vec2( 9.082951, 196.8), vec2( 9.137833, 199.2),
    vec2( 9.192388, 201.6), vec2( 9.246621, 204.0), vec2( 9.300538, 206.4), vec2( 9.354143, 208.8),
    vec2( 9.407444, 211.2), vec2( 9.460444, 213.6), vec2( 9.513149, 216.0), vec2( 9.565563, 218.4),
    vec2( 9.617692, 220.8), vec2( 9.669540, 223.2), vec2( 9.721111, 225.6), vec2( 9.772410, 228.0),
    vec2( 9.823441, 230.4), vec2( 9.874209, 232.8), vec2( 9.924717, 235.2), vec2( 9.974969, 237.6),
    vec2(10.024969, 240.0), vec2(10.074721, 242.4), vec2(10.124228, 244.8), vec2(10.173495, 247.2),
    vec2(10.222524, 249.6), vec2(10.271319, 252.0), vec2(10.319884, 254.4), vec2(10.368221, 256.8),
    vec2(10.416333, 259.2), vec2(10.464225, 261.6), vec2(10.511898, 264.0), vec2(10.559356, 266.4),
    vec2(10.606602, 268.8), vec2(10.653638, 271.2), vec2(10.700467, 273.6), vec2(10.747093, 276.0),
    vec2(10.793517, 278.4), vec2(10.839742, 280.8), vec2(10.885771, 283.2), vec2(10.931606, 285.6),
    vec2(10.977249, 288.0), vec2(11.022704, 290.4), vec2(11.067972, 292.8), vec2(11.113055, 295.2),
    vec2(11.157957, 297.6), vec2(11.202678, 300.0), vec2(11.247222, 302.4), vec2(11.291590, 304.8)
);

uniform vec3 u_LightPos[MAX_LIGHTS];
uniform int u_NumLights;
uniform int u_PointLightCastsShadow[MAX_LIGHTS];
uniform samplerCube u_ShadowMap[MAX_LIGHTS];
uniform float u_FarPlane[MAX_LIGHTS];

uniform vec3 u_DirLightDirection[MAX_LIGHTS];
uniform int u_NumDirLights;
uniform int u_DirLightCastsShadow[MAX_LIGHTS];
uniform sampler2DArray u_DirShadowMap[MAX_LIGHTS];
uniform mat4 u_DirLightSpaceMatrices[MAX_LIGHTS * 3];
uniform float u_CascadeDistances[3];

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

float calculateDirectionalShadow(vec4 fragPosLightSpace, sampler2DArray shadowMap, float cascadeIndex, float bias, float randomRotation) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    if (projCoords.z > 1.0) return 0.0;

    float currentDepth = projCoords.z;
    float shadow = 0.0;
    for (int i = 0; i < u_DirPcfSamples; ++i) {
        float r = VOGEL_DISK[i].x * u_InvSqrtDirPcfSamples;
        float theta = VOGEL_DISK[i].y + randomRotation;
        vec2 offset = vec2(cos(theta), sin(theta)) * r * 0.001;

        float closestDepth = texture(shadowMap, vec3(projCoords.xy + offset, cascadeIndex)).r;
        if (currentDepth - bias > closestDepth) shadow += 1.0;
    }

    return pow(shadow * u_InvDirPcfSamples, 2.4);
}

void main() {
    float depth = texture(u_DepthTexture, v_UV).r;
    if (depth == 1.0) discard;

    vec3 N = normalize(texture(u_NormalTexture, v_UV).xyz * 2.0 - 1.0);

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
                float normalOffsetScale = max(0.006 * (1.0 - dot(N, L)), 0.0006);
                vec3 biasedWorldPos = v_WorldPos + N * normalOffsetScale;
                float shadow = calculatePointShadow(
                    biasedWorldPos, u_LightPos[i], u_ShadowMap[i],
                    u_FarPlane[i], 0.0001, randomRotation
                );
                pointShadows[i] = smoothstep(0.02, 0.98, shadow);
            }
        }
    }

    vec4 viewSpacePos = u_View * vec4(v_WorldPos, 1.0);
    float viewSpaceZ = -viewSpacePos.z;

    vec4 dirShadows = vec4(0.0);
    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= u_NumDirLights) break;
        if (u_DirLightCastsShadow[i] == 1) {
            vec3 L = normalize(-u_DirLightDirection[i]);
            float normalOffsetScale = max(0.002 * (1.0 - dot(N, L)), 0.0002);

            int cascadeIndex = 2;
            if (viewSpaceZ <= u_CascadeDistances[0]) cascadeIndex = 0;
            else if (viewSpaceZ <= u_CascadeDistances[1]) cascadeIndex = 1;

            // now, the required world bias amount doesn't scale like this,
            // but actually scaling the appropriate amount will result in severe peter panning.
            // we scale it just a bit to squash the risk of it appearing even with PCF sample count >= 16.
            // generally PCF will naturally filter it out enough.
            float cascadeScaleFactor = float(cascadeIndex + 1) * 4.0;
            vec3 biasedWorldPos = v_WorldPos + N * normalOffsetScale * cascadeScaleFactor;
            vec4 fragPosLightSpace = u_DirLightSpaceMatrices[i * 3 + cascadeIndex] * vec4(biasedWorldPos, 1.0);

            float shadow = calculateDirectionalShadow(
                fragPosLightSpace, u_DirShadowMap[i], float(cascadeIndex), 0.0001, randomRotation
            );

            if (cascadeIndex < 2) {
                float nextSplit = u_CascadeDistances[cascadeIndex];
                float prevSplit = cascadeIndex == 0 ? 0.0 : u_CascadeDistances[cascadeIndex - 1];
                float splitSize = nextSplit - prevSplit;
                float blendBand = splitSize * 0.05;

                if (viewSpaceZ > nextSplit - blendBand) {
                    float blendFactor = (viewSpaceZ - (nextSplit - blendBand)) / blendBand;

                    float innerCascadeScaleFactor = float(cascadeIndex + 2) * 4.0;
                    vec3 innerBiasedWorldPos = v_WorldPos + N * normalOffsetScale * innerCascadeScaleFactor;
                    vec4 nextFragPosLightSpace = u_DirLightSpaceMatrices[i * 3 + cascadeIndex + 1] * vec4(innerBiasedWorldPos, 1.0);
                    float nextShadow = calculateDirectionalShadow(
                        nextFragPosLightSpace, u_DirShadowMap[i], float(cascadeIndex + 1), 0.0001, randomRotation
                    );

                    shadow = mix(shadow, nextShadow, blendFactor);
                }
            }

            dirShadows[i] = smoothstep(0.02, 0.98, shadow);
        }
    }

    out_PointShadows = pointShadows;
    out_DirShadows = dirShadows;
}
