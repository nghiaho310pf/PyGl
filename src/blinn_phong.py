from shader import Shader

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 a_Pos;
layout (location = 1) in vec3 a_Normal;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;

out vec3 v_Normal;
out vec3 v_FragPos;

void main() {
    v_FragPos = vec3(u_Model * vec4(a_Pos, 1.0));
    v_Normal = mat3(transpose(inverse(u_Model))) * a_Normal; 
    gl_Position = u_Projection * u_View * vec4(v_FragPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

in vec3 v_Normal;
in vec3 v_FragPos;

uniform vec3 u_Color;      // Material Albedo
uniform vec3 u_LightPos;   
uniform vec3 u_ViewPos;

// Parameters (Later these can be material properties)
float specularStrength = 0.5;
float shininess = 32.0; 

void main() {
    // 1. Geometry Data
    vec3 normal = normalize(v_Normal);
    vec3 lightDir = normalize(u_LightPos - v_FragPos);
    vec3 viewDir = normalize(u_ViewPos - v_FragPos);
    
    // 2. Ambient (Soft constant light so shadows aren't pitch black)
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * vec3(1.0); // White ambient light
  
    // 3. Diffuse (Lambertian)
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0); // White light color

    // 4. Specular (Blinn-Phong)
    // The "Halfway Vector" is the secret sauce. 
    // It creates a softer, more realistic highlight than Phong.
    vec3 halfwayDir = normalize(lightDir + viewDir);  
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * vec3(1.0); 
    
    // 5. Combine
    // Note: We multiply the light components by the Albedo (u_Color)
    vec3 result = (ambient + diffuse + specular) * u_Color;

    // --- KEY STEP: GAMMA CORRECTION ---
    // Monitor expects sRGB, but we calculated in Linear.
    // We must apply Gamma correction (power of 1/2.2) to brighten the shadows correctly.
    result = pow(result, vec3(1.0 / 2.2)); 

    FragColor = vec4(result, 1.0);
}
"""


def make_shader():
    return Shader(VERTEX_SHADER, FRAGMENT_SHADER)
