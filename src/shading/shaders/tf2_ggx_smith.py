from shading.shader import Shader

VERTEX_SHADER = """
#version 450 core
layout (location = 0) in vec3 a_Pos;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_UV;

layout (std140) uniform SceneData {
    mat4 u_Projection;
    mat4 u_View;
    vec3 u_ViewPos;
    float u_Time;
};

uniform mat4 u_Model;

out vec3 v_WorldPos;
out vec3 v_Normal;
out vec2 v_UV;

void main() {
    v_WorldPos = vec3(u_Model * vec4(a_Pos, 1.0));
    // Normal Matrix to handle non-uniform scaling correctly
    v_Normal = mat3(transpose(inverse(u_Model))) * a_Normal; 
    v_UV = a_UV;
    
    gl_Position = u_Projection * u_View * vec4(v_WorldPos, 1.0);
}
"""

FRAGMENT_SHADER = """
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
uniform samplerCube u_ShadowMap[MAX_LIGHTS];
uniform float u_FarPlane[MAX_LIGHTS];

const float PI = 3.14159265359;

float filmGrain(vec2 coords) {
    return fract(sin(dot(coords.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / max(denom, 0.0001);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / max(denom, 0.0001);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Hammon diffuse, replaces the standard Lambert (Albedo / PI).
// accounts for roughness at grazing angles (retro-reflection) and 
// improves energy conservation compared to pure lambert
float HammonDiffuse(vec3 N, vec3 V, vec3 L, float roughness) {
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

vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

vec3 sampleOffsetDirections[20] = vec3[]
(
   vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1),
   vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
   vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
   vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
   vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
);

float FindBlockerDistance(samplerCube shadowMap, vec3 fragToLight, float currentDepth, float farPlane) {
    int blockers = 0;
    float avgBlockerDepth = 0.0;
    float searchWidth = 0.05;

    int samples = 8;
    for (int i = 0; i < samples; ++i) {
        vec3 sampleDir = fragToLight + sampleOffsetDirections[i] * searchWidth;
        float sampleDepth = texture(shadowMap, sampleDir).r;

        if (sampleDepth < currentDepth - 0.005) {
            avgBlockerDepth += sampleDepth;
            blockers++;
        }
    }

    if (blockers > 0)
        return avgBlockerDepth / float(blockers);
    
    return -1.0;
}

float ShadowCalculation(vec3 fragPos, vec3 lightPos, float lightRadius, samplerCube shadowMap, float farPlane) {
    vec3 fragToLight = fragPos - lightPos;
    float currentDepth = length(fragToLight) / farPlane;

    float bias = 0.005;

    float blockerDepth = FindBlockerDistance(shadowMap, fragToLight, currentDepth, farPlane);
    if (blockerDepth == -1.0) return 0.0;

    float penumbraRatio = (currentDepth - blockerDepth) / blockerDepth;
    float diskRadius = penumbraRatio * lightRadius; 

    diskRadius = clamp(diskRadius, 0.001, 0.15);

    float shadow = 0.0;
    int samples = 20;

    for (int i = 0; i < samples; ++i) {
        vec3 sampleDir = fragToLight + sampleOffsetDirections[i] * diskRadius;
        float closestDepth = texture(shadowMap, sampleDir).r;

        if (currentDepth - bias > closestDepth)
            shadow += 1.0;
    }
    shadow /= float(samples);

    return shadow;
}

void main() {
    vec3 N = normalize(v_Normal);
    vec3 V = normalize(u_ViewPos - v_WorldPos);

    vec3 ambient = u_Albedo * u_AO;
    vec3 totalDirectLight = vec3(0.0);

    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= u_NumLights) {
            break;
        }

        vec3 lightPos = u_LightPos[i];
        vec3 lightColor = u_LightColor[i];

        vec3 L = normalize(lightPos - v_WorldPos);
        vec3 H = normalize(V + L);
        float distance = length(lightPos - v_WorldPos);

        // Shadow calculation
        float shadow = ShadowCalculation(v_WorldPos, lightPos, u_LightRadius[i], u_ShadowMap[i], u_FarPlane[i]);

        // lighting prep
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColor * attenuation;

        // F0. remap user reflectance (0.0-1.0) to physical dielectric range (0.0-0.08)
        // 0.5 input becomes 0.04 (standard plastic/water)
        // 1.0 input becomes 0.08 (gemstone/high-gloss)
        float dielectricF0 = 0.16 * u_Reflectance * u_Reflectance;
        vec3 F0 = vec3(dielectricF0); 
        F0 = mix(F0, u_Albedo, u_Metallic);

        // cook-torrance specular
        float NDF = DistributionGGX(N, H, u_Roughness);
        float G = GeometrySmith(N, V, L, u_Roughness);
        vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; 
        vec3 specular = numerator / denominator;

        // Hammon diffuse
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= (1.0 - u_Metallic);

        // Hammon's roughness-dependent diffuse
        float diffuseBRDF = HammonDiffuse(N, V, L, u_Roughness);

        // translucency; wraps the light around the terminator
        float wrap = u_Translucency * 0.5; // Scale down for valid range
        float NdotL_Unclamped = dot(N, L);
        float NdotL_Wrapped = max((NdotL_Unclamped + wrap) / (1.0 + wrap), 0.0);

        // combine diffuse. multiply by PI because HammonDiffuse already divides by PI
        vec3 diffuse = kD * u_Albedo * diffuseBRDF;

        // composite direct light. specular relies on pure NdotL, diffuse relies on wrapped NdotL
        // we add u_AO to diffuse attenuation slightly to fake self-shadowing in cracks
        vec3 directLight = (diffuse * NdotL_Wrapped + specular * max(dot(N, L), 0.0)) * radiance;

        // apply shadow
        directLight *= (1.0 - shadow);

        // specular occlusion
        // trick to reduce specular intensity in cracks (where AO is low). prevents "shining in the dark"
        float specularOcclusion = clamp(pow(NdotL_Wrapped + u_AO, 2.0), 0.0, 1.0);
        // directLight *= specularOcclusion; // optional: apply to direct light too? Usually just ambient.

        totalDirectLight += directLight;
    }
    
    vec3 color = ambient + totalDirectLight;

    // post-processing
    // Reinhard HDR tonemapping
    color = ACESFilm(color);
    // dithering
    color += (filmGrain(gl_FragCoord.xy + fract(u_Time)) - 0.5) / 255.0;

    FragColor = vec4(color, 1.0);
}
"""


def make_shader():
    return Shader(VERTEX_SHADER, FRAGMENT_SHADER)
