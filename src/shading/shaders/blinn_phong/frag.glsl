#version 450 core
out vec4 FragColor;

in vec3 v_Normal;
in vec3 v_FragPos;

layout (std140) uniform SceneData {
    mat4 u_Projection;
    mat4 u_View;
    vec3 u_ViewPos;
    float u_Time;
};

uniform mat4 u_Model;
uniform vec3 u_Albedo;
uniform float u_Roughness;
uniform float u_Reflectance;
uniform float u_AO;

const int MAX_LIGHTS = 4;
uniform vec3 u_LightPos[MAX_LIGHTS];
uniform vec3 u_LightColor[MAX_LIGHTS];
uniform int u_NumLights;

float filmGrain(vec2 coords) {
    return fract(sin(dot(coords.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec3 normal = normalize(v_Normal);
    vec3 viewDir = normalize(u_ViewPos - v_FragPos);
    vec3 ambient = vec3(u_AO);

    float shininess = 2.0 / pow(max(u_Roughness, 0.0001), 4.0) - 2.0;
    float actualReflectance = 0.16 * (u_Reflectance * u_Reflectance);
    float energyConservation = (shininess + 8.0) * 0.039789;
    float specMultiplier = actualReflectance * energyConservation;

    vec3 diffuseAccumulator = vec3(0.0);
    vec3 specularAccumulator = vec3(0.0);

    int numLights = min(u_NumLights, MAX_LIGHTS); 
    for (int i = 0; i < numLights; i++) {
        vec3 lightDir = normalize(u_LightPos[i] - v_FragPos);
        vec3 halfwayDir = normalize(lightDir + viewDir);

        float NdotL = max(dot(normal, lightDir), 0.0);
        float NdotH = max(dot(normal, halfwayDir), 0.0);

        diffuseAccumulator += NdotL * u_LightColor[i];

        float specAmount = pow(NdotH, shininess);
        specularAccumulator += specAmount * u_LightColor[i];
    }

    diffuseAccumulator *= 0.0033;
    specularAccumulator *= specMultiplier;
    vec3 result = ((ambient + diffuseAccumulator) * u_Albedo) + specularAccumulator;

    result += (filmGrain(gl_FragCoord.xy + fract(u_Time)) - 0.5) * 0.002;

    FragColor = vec4(result, 1.0);
}