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
uniform vec3 u_LightPos;
uniform vec3 u_LightColor;

float specularStrength = 0.5;
float shininess = 32.0; 

void main() {
    vec3 normal = normalize(v_Normal);
    vec3 lightDir = normalize(u_LightPos - v_FragPos);
    vec3 viewDir = normalize(u_ViewPos - v_FragPos);

    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * vec3(1.0);

    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * u_LightColor * 0.001;

    vec3 halfwayDir = normalize(lightDir + viewDir);  
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * vec3(1.0); 

    vec3 result = (ambient + diffuse + specular) * u_Albedo;

    FragColor = vec4(result, 1.0);
}