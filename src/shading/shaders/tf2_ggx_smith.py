from shading.shader import Shader

VERTEX_SHADER = """
#version 330 core
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
#version 330 core
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

uniform vec3 u_LightPos;
uniform vec3 u_LightColor;

const float PI = 3.14159265359;

float random(vec2 coords) {
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

void main() {
    vec3 N = normalize(v_Normal);
    vec3 V = normalize(u_ViewPos - v_WorldPos);
    vec3 L = normalize(u_LightPos - v_WorldPos);
    vec3 H = normalize(V + L);
    float distance = length(u_LightPos - v_WorldPos);

    // lighting prep
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = u_LightColor * attenuation;

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

    // ambient / ibm approx
    // in real PBR engine, this is replaced by an irradiance map and prefiltered env map
    vec3 ambient = vec3(0.03) * u_Albedo * u_AO;

    // specular occlusion
    // trick to reduce specular intensity in cracks (where AO is low). prevents "shining in the dark"
    float specularOcclusion = clamp(pow(NdotL_Wrapped + u_AO, 2.0), 0.0, 1.0);
    // directLight *= specularOcclusion; // optional: apply to direct light too? Usually just ambient.

    vec3 color = ambient + directLight;

    // post-processing
    // Reinhard HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correction
    color = pow(color, vec3(1.0/2.2));
    // dithering
    color += (random(gl_FragCoord.xy + fract(u_Time)) - 0.5) / 255.0;

    FragColor = vec4(color, 1.0);
}
"""


def make_shader():
    return Shader(VERTEX_SHADER, FRAGMENT_SHADER)
