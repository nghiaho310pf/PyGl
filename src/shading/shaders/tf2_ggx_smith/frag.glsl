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
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
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
    // 1. AgX Input Transform (Inset)
    // Moves Rec.709 primaries inward to avoid clipping "The Notorious Six"
    const mat3 agx_input_mat = mat3(
        0.59719, 0.35458, 0.04823,
        0.07600, 0.90834, 0.01566,
        0.02840, 0.13383, 0.83777
    );

    vec3 val = agx_input_mat * color;

    // 2. Log2 Space Encoding
    const float min_ev = -12.47393;
    const float max_ev = 4.026069;
    val = clamp(log2(val), min_ev, max_ev);
    val = (val - min_ev) / (max_ev - min_ev);

    // 3. Sigmoid (Approximation of the standard AgX curve)
    // Generates the "S" curve for contrast
    val = clamp(val, 0.0, 1.0);
    vec3 val2 = val * val;
    vec3 val4 = val2 * val2;
    val = -17.86  * val  * val2 * val4
        + 79.05   * val4 * val2
        - 129.56  * val4 * val
        + 90.30   * val4
        - 22.56   * val2 * val
        + 1.34    * val2;

    // 4. AgX Output Transform (Outset)
    // Convert back to Linear Rec.709
    const mat3 agx_output_mat = mat3(
        1.56230, -0.46872, -0.09358,
        -0.45667, 1.35334, 0.10332,
        -0.09117, 0.02988, 1.06129
    );

    return agx_output_mat * val;
}

vec3 khronosPbrNeutral(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    const float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, newPeak * vec3(1.0, 1.0, 1.0), g);
}

vec3 sampleOffsetDirections[20] = vec3[]
(
   vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1),
   vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
   vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
   vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
   vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
);

float blueNoiseDither(vec2 pos) {
    float white = fract(sin(dot(pos, vec2(12.9898, 78.233))) * 43758.5453);
    float noise = fract(white + (gl_FragCoord.x + gl_FragCoord.y * 0.5) * 0.375);
    return noise;
}

vec3 sampleVogelDisk(int sampleIndex, int samplesCount, float rotation) {
    float goldenAngle = 2.4; // approx PI * (3.0 - sqrt(5.0))

    float r = sqrt(float(sampleIndex) + 0.5) / sqrt(float(samplesCount));
    float theta = float(sampleIndex) * goldenAngle + rotation;

    float sine = sin(theta);
    float cosine = cos(theta);

    return vec3(r * cosine, r * sine, 0.0);
}

float calculateShadow(vec3 fragPos, vec3 lightPos, float lightRadius, samplerCube shadowMap, float farPlane, float bias) {
    vec3 fragToLight = fragPos - lightPos;
    float currentDepth = length(fragToLight) / farPlane;

    float noise = blueNoiseDither(gl_FragCoord.xy);
    float randomRotation = noise * PI2;

    int searchSamples = 8;
    float avgBlockerDepth = 0.0;
    int blockers = 0;
    float searchWidth = 0.05 + (currentDepth * 0.01); // Search wider further away

    vec3 lightDir = normalize(fragToLight);
    vec3 up = abs(lightDir.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(up, lightDir));
    up = cross(lightDir, right);

    for (int i = 0; i < searchSamples; ++i) {
        vec3 offset2D = sampleVogelDisk(i, searchSamples, randomRotation);
        vec3 sampleDir = lightDir + (right * offset2D.x + up * offset2D.y) * searchWidth;

        float sampleDepth = texture(shadowMap, sampleDir).r;
        if (sampleDepth < currentDepth - bias) {
            avgBlockerDepth += sampleDepth;
            blockers++;
        }
    }

    if (blockers == 0) return 0.0;
    avgBlockerDepth /= float(blockers);

    float penumbraRatio = (currentDepth - avgBlockerDepth) / avgBlockerDepth;
    float diskRadius = penumbraRatio * lightRadius;
    diskRadius = clamp(diskRadius, 0.002, 0.15);

    float shadow = 0.0;
    int pcfSamples = 32;

    float invSamples = 1.0 / float(pcfSamples);

    for (int i = 0; i < pcfSamples; ++i) {
        vec3 offset2D = sampleVogelDisk(i, pcfSamples, randomRotation);

        float jitter = fract(noise + u_Time + float(i) * 0.618);
        float radiusJitter = 1.0 + (jitter * 0.2 - 0.1);
        vec3 sampleDir = lightDir + (right * offset2D.x + up * offset2D.y) * (diskRadius * radiusJitter);

        float closestDepth = texture(shadowMap, sampleDir).r;
        if (currentDepth - bias > closestDepth) shadow += 1.0;
    }

    return shadow * invSamples;
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
        float shadowBias = max(0.0075 * (1.0 - dot(N, L)), 0.00075);
        float shadow = calculateShadow(v_WorldPos, lightPos, u_LightRadius[i], u_ShadowMap[i], u_FarPlane[i], shadowBias);
        shadow = smoothstep(0.01, 0.98, shadow);

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

    // tonemapping
    color = khronosPbrNeutral(color);
    // dithering
    color += (filmGrain(gl_FragCoord.xy + fract(u_Time)) - 0.5) / 255.0;
    FragColor = vec4(color, 1.0);
}