#version 450 core
out vec4 FragColor;

in vec3 v_WorldPos;
in vec3 v_Normal;
in vec2 v_UV;

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
uniform int u_PointLightCastsShadow[MAX_LIGHTS];
uniform float u_FarPlane[MAX_LIGHTS];

uniform vec3 u_DirLightDirection[MAX_LIGHTS];
uniform vec3 u_DirLightColor[MAX_LIGHTS];
uniform int u_NumDirLights;
uniform int u_DirLightCastsShadow[MAX_LIGHTS];

uniform sampler2D u_AlbedoMap;
uniform bool u_UseAlbedoMap;

uniform sampler2D u_PointShadowMask;
uniform sampler2D u_DirShadowMask;
uniform vec2 u_ScreenSize;

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

float hammonDiffuse(vec3 N, vec3 V, vec3 L, float roughness) {
    vec3 H = normalize(V + L);
    float LdotH = clamp(dot(L, H), 0.0, 1.0);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float NdotV = clamp(dot(N, V), 0.0, 1.0);

    float facing = 0.5 + 2.0 * LdotH * LdotH * roughness;
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

void main() {
    vec3 N = normalize(v_Normal);
    if (!gl_FrontFacing) {
        N = -N;
    }

    vec3 V = normalize(u_ViewPos - v_WorldPos);

    vec3 finalAlbedo = u_Albedo;
    if (u_UseAlbedoMap) {
        vec4 albedoTex = texture(u_AlbedoMap, v_UV);
        finalAlbedo *= albedoTex.xyz;
    }

    vec3 ambient = finalAlbedo * u_AO;
    vec3 totalDirectLight = vec3(0.0);

    vec2 screenUV = gl_FragCoord.xy / u_ScreenSize;
    vec4 pointShadowMask = texture(u_PointShadowMask, screenUV);
    vec4 dirShadowMask = texture(u_DirShadowMask, screenUV);

    // == point lights ==
    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= u_NumLights) break;

        vec3 lightPos = u_LightPos[i];
        float distance = length(lightPos - v_WorldPos);
        if (distance > u_FarPlane[i]) continue;

        vec3 lightColor = u_LightColor[i];
        vec3 L = normalize(lightPos - v_WorldPos);
        vec3 H = normalize(V + L);

        float shadow = pointShadowMask[i];

        // lighting prep
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColor * attenuation;

        float dielectricF0 = 0.16 * u_Reflectance * u_Reflectance;
        vec3 F0 = vec3(dielectricF0);
        F0 = mix(F0, finalAlbedo, u_Metallic);

        float NDF = distributionGGX(N, H, u_Roughness);
        float G = correlatedSmith(dot(N, V), dot(N, L), u_Roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 specular = NDF * G * F;

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= (1.0 - u_Metallic);

        float diffuseBRDF = hammonDiffuse(N, V, L, u_Roughness);

        float wrap = u_Translucency * 0.5;
        float NdotL_Unclamped = dot(N, L);
        float NdotL_Wrapped = max((NdotL_Unclamped + wrap) / (1.0 + wrap), 0.0);

        vec3 diffuse = kD * finalAlbedo * diffuseBRDF;
        vec3 directLight = (diffuse * NdotL_Wrapped + specular * max(dot(N, L), 0.0)) * radiance;

        directLight *= (1.0 - shadow);
        totalDirectLight += directLight;
    }

    // == directional lights ==
    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= u_NumDirLights) break;

        vec3 lightDir = normalize(-u_DirLightDirection[i]);
        vec3 lightColor = u_DirLightColor[i];

        vec3 L = lightDir;
        vec3 H = normalize(V + L);

        float shadow = dirShadowMask[i];

        vec3 radiance = lightColor;

        float dielectricF0 = 0.16 * u_Reflectance * u_Reflectance;
        vec3 F0 = vec3(dielectricF0);
        F0 = mix(F0, finalAlbedo, u_Metallic);

        float NDF = distributionGGX(N, H, u_Roughness);
        float G = correlatedSmith(dot(N, V), dot(N, L), u_Roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 specular = NDF * G * F;

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= (1.0 - u_Metallic);

        float diffuseBRDF = hammonDiffuse(N, V, L, u_Roughness);

        float wrap = u_Translucency * 0.5;
        float NdotL_Unclamped = dot(N, L);
        float NdotL_Wrapped = max((NdotL_Unclamped + wrap) / (1.0 + wrap), 0.0);

        vec3 diffuse = kD * finalAlbedo * diffuseBRDF;
        vec3 directLight = (diffuse * NdotL_Wrapped + specular * max(dot(N, L), 0.0)) * radiance;

        directLight *= (1.0 - shadow);
        totalDirectLight += directLight;
    }

    vec3 color = ambient + totalDirectLight;
    color = toneMapAgX(color);
    color += (filmGrain(gl_FragCoord.xy + fract(u_Time)) - 0.5) * 0.002;
    FragColor = vec4(color, 1.0);
}
