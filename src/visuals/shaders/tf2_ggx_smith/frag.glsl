#version 450 core
out vec4 FragColor;

centroid in vec3 v_WorldPos;
centroid in vec3 v_Normal;
centroid in vec2 v_UV;

layout (std140) uniform SceneData {
    mat4 u_Projection;
    mat4 u_View;
    vec3 u_ViewPos;
    float u_Time;
};

uniform vec3 u_Albedo;
uniform float u_Roughness;
uniform float u_Metallic;

// Hammon parameters
uniform float u_Reflectance;   // F0 control (0.5 = 0.04 standard dielectric)
uniform float u_Translucency;  // "Wrap" factor for subsurface scattering
uniform float u_AO;            // Ambient Occlusion

const int MAX_LIGHTS = 4;
uniform vec3 u_LightPos[MAX_LIGHTS];
uniform float u_LightRadius[MAX_LIGHTS];
uniform vec3 u_LightColor[MAX_LIGHTS];
uniform int u_NumLights;
uniform samplerCube u_ShadowMap[MAX_LIGHTS];
uniform float u_FarPlane[MAX_LIGHTS];

uniform vec3 u_DirLightDirection[MAX_LIGHTS];
uniform vec3 u_DirLightColor[MAX_LIGHTS];
uniform int u_NumDirLights;
uniform sampler2D u_DirShadowMap[MAX_LIGHTS];
uniform mat4 u_DirLightSpaceMatrix[MAX_LIGHTS];

uniform sampler2D u_AlbedoMap;
uniform int u_UseAlbedoMap;

const float PI = 3.14159265359;
const float PI2 = 6.28318530718;

float filmGrain(vec2 coords) {
    return fract(sin(dot(coords.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / max(denom, 0.0001);
}

float correlatedSmith(float NdotV, float NdotL, float roughness) {
    float a2 = roughness * roughness;
    float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / (GGXV + GGXL);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    float f = clamp(1.0 - cosTheta, 0.0, 1.0);
    float f2 = f * f;
    float f5 = f2 * f2 * f; 
    return F0 + (1.0 - F0) * f5;
}

// Hammon diffuse, replaces the standard Lambert (Albedo / PI).
// accounts for roughness at grazing angles (retro-reflection) and
// improves energy conservation compared to pure lambert
float hammonDiffuse(vec3 N, vec3 V, vec3 L, float roughness) {
    vec3 H = normalize(V + L);
    float LdotH = clamp(dot(L, H), 0.0, 1.0);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float NdotV = clamp(dot(N, V), 0.0, 1.0);

    // Disney/Hammon diffuse factor approximation
    // 0.5 is the base, LdotH * LdotH simulates the retro-reflective peak
    float facing = 0.5 + 2.0 * LdotH * LdotH * roughness;

    // smooth out the grazing angles
    return facing / PI;
}

vec3 toneMapAgX(vec3 color) {
    const mat3 agx_input_mat = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );

    vec3 val = agx_input_mat * color;

    val = clamp(log2(max(val, 1e-6)) * 0.0606060606 + 0.7559957575, 0.0, 1.0);

    vec3 x2 = val * val;
    vec3 x4 = x2 * x2;
    val = 15.5 * x4 * x2 - 40.14 * x4 * val + 31.96 * x4 - 6.868 * x2 * val + 0.4298 * x2 + 0.1191 * val - 0.00232;

    const mat3 agx_output_mat = mat3(
        1.56230, -0.46872, -0.09358,
        -0.45667, 1.35334, 0.10332,
        -0.09117, 0.02988, 1.06129
    );

    return pow(max(agx_output_mat * val, 0.0), vec3(2.2));
}

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

    // precalculate rotation matrix and initial vector
    const float GOLDEN_ANGLE = 2.4;
    const mat2 vogelRot = mat2(cos(GOLDEN_ANGLE), sin(GOLDEN_ANGLE), -sin(GOLDEN_ANGLE), cos(GOLDEN_ANGLE));
    
    vec2 vogelDir = vec2(cos(randomRotation), sin(randomRotation));
    float invSqrtSearchSamples = 1.0 / sqrt(float(searchSamples));

    for (int i = 0; i < searchSamples; ++i) {
        float r = sqrt(float(i) + 0.5) * invSqrtSearchSamples;
        vec2 offset2D = vogelDir * r;
        vogelDir = vogelRot * vogelDir; // re-rotate

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

    vogelDir = vec2(cos(randomRotation), sin(randomRotation)); // reset rotation

    for (int i = 0; i < pcfSamples; ++i) {
        float r = sqrt(float(i) + 0.5) * invSqrtPcfSamples;
        vec2 offset2D = vogelDir * r;
        vogelDir = vogelRot * vogelDir; // re-rotate

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
    
    vogelDir = vec2(cos(randomRotation), sin(randomRotation)); // reset

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
    vec3 N = normalize(v_Normal);
    vec3 V = normalize(u_ViewPos - v_WorldPos);

    vec3 finalAlbedo = u_Albedo;
    if (u_UseAlbedoMap == 1) {
        vec4 albedoTex = texture(u_AlbedoMap, v_UV);
        finalAlbedo *= albedoTex.xyz;
    }

    vec3 ambient = finalAlbedo * u_AO;
    vec3 totalDirectLight = vec3(0.0);

    float globalNoise = blueNoiseDither(gl_FragCoord.xy * fract(u_Time * 2.5));
    float randomRotation = globalNoise * PI2;

    // == point lights ==
    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= u_NumLights) {
            break;
        }

        vec3 lightPos = u_LightPos[i];
        float distance = length(lightPos - v_WorldPos);
        if (distance > u_FarPlane[i]) {
            continue;
        }

        vec3 lightColor = u_LightColor[i];

        vec3 L = normalize(lightPos - v_WorldPos);
        vec3 H = normalize(V + L);

        float normalOffsetScale = max(0.02 * (1.0 - dot(N, L)), 0.002);
        vec3 biasedWorldPos = v_WorldPos + N * normalOffsetScale;
        float constantDepthBias = 0.00025;
        float shadow = calculatePointShadow(
            biasedWorldPos,
            lightPos,
            u_LightRadius[i],
            u_ShadowMap[i],
            u_FarPlane[i],
            constantDepthBias,
            randomRotation 
        );
        shadow = smoothstep(0.01, 0.98, shadow);

        // lighting prep
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColor * attenuation;

        // F0. remap user reflectance (0.0-1.0) to physical dielectric range (0.0-0.08)
        // 0.5 input becomes 0.04 (standard plastic/water)
        // 1.0 input becomes 0.08 (gemstone/high-gloss)
        float dielectricF0 = 0.16 * u_Reflectance * u_Reflectance;
        vec3 F0 = vec3(dielectricF0);
        F0 = mix(F0, finalAlbedo, u_Metallic);

        // cook-torrance specular
        float NDF = distributionGGX(N, H, u_Roughness);
        float G = correlatedSmith(dot(N, V), dot(N, L), u_Roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 specular = NDF * G * F;

        // Hammon diffuse
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= (1.0 - u_Metallic);

        // Hammon's roughness-dependent diffuse
        float diffuseBRDF = hammonDiffuse(N, V, L, u_Roughness);

        // translucency; wraps the light around the terminator
        float wrap = u_Translucency * 0.5; // Scale down for valid range
        float NdotL_Unclamped = dot(N, L);
        float NdotL_Wrapped = max((NdotL_Unclamped + wrap) / (1.0 + wrap), 0.0);

        // combine diffuse. multiply by PI because hammonDiffuse already divides by PI
        vec3 diffuse = kD * finalAlbedo * diffuseBRDF;

        // composite direct light. specular relies on pure NdotL, diffuse relies on wrapped NdotL
        // we add u_AO to diffuse attenuation slightly to fake self-shadowing in cracks
        vec3 directLight = (diffuse * NdotL_Wrapped + specular * max(dot(N, L), 0.0)) * radiance;

        // apply shadow
        directLight *= (1.0 - shadow);

        totalDirectLight += directLight;
    }

    // == directional lights ==
    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= u_NumDirLights) {
            break;
        }

        vec3 lightDir = normalize(-u_DirLightDirection[i]);
        vec3 lightColor = u_DirLightColor[i];

        vec3 L = lightDir;
        vec3 H = normalize(V + L);

        float normalOffsetScale = max(0.02 * (1.0 - dot(N, L)), 0.002);
        vec3 biasedWorldPos = v_WorldPos + N * normalOffsetScale;
        vec4 fragPosLightSpace = u_DirLightSpaceMatrix[i] * vec4(biasedWorldPos, 1.0);
        float shadow = calculateDirectionalShadow(
            fragPosLightSpace,
            u_DirShadowMap[i],
            0.0002,
            randomRotation
        );

        // lighting prep
        vec3 radiance = lightColor; // no attenuation for directional lights

        // F0
        float dielectricF0 = 0.16 * u_Reflectance * u_Reflectance;
        vec3 F0 = vec3(dielectricF0);
        F0 = mix(F0, finalAlbedo, u_Metallic);

        // cook-torrance specular
        float NDF = distributionGGX(N, H, u_Roughness);
        float G = correlatedSmith(dot(N, V), dot(N, L), u_Roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 specular = NDF * G * F;

        // Hammon diffuse
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= (1.0 - u_Metallic);

        float diffuseBRDF = hammonDiffuse(N, V, L, u_Roughness);

        float wrap = u_Translucency * 0.5;
        float NdotL_Unclamped = dot(N, L);
        float NdotL_Wrapped = max((NdotL_Unclamped + wrap) / (1.0 + wrap), 0.0);

        vec3 diffuse = kD * finalAlbedo * diffuseBRDF;

        vec3 directLight = (diffuse * NdotL_Wrapped + specular * max(dot(N, L), 0.0)) * radiance;

        // apply shadow
        directLight *= (1.0 - shadow);

        totalDirectLight += directLight;
    }

    vec3 color = ambient + totalDirectLight;

    // tonemapping
    color = toneMapAgX(color);
    // dithering
    color += (filmGrain(gl_FragCoord.xy + fract(u_Time)) - 0.5) * 0.002;
    FragColor = vec4(color, 1.0);
}