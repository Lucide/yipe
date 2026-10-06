#version 330

// Uncomment this line to inspect the interpolated normals as RGB colors.
// #define DEBUG_NORMAL_COLOR

in vec3 worldPosition;
in vec3 worldNormal;

uniform vec3 viewPos;
uniform vec4 colDiffuse;

out vec4 finalColor;

void main() {
    vec3 normal = normalize(worldNormal);

#ifdef DEBUG_NORMAL_COLOR
    finalColor = vec4(normal * 0.5 + 0.5, 1.0);
    return;
#endif

    vec3 lightDirection = normalize(vec3(-0.45, 0.8, 0.35));
    vec3 viewDirection = normalize(viewPos - worldPosition);
    vec3 halfDirection = normalize(lightDirection + viewDirection);

    float diffuse = max(dot(normal, lightDirection), 0.0);
    float specular = pow(max(dot(normal, halfDirection), 0.0), 64.0);
    float rim = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.0);

    // A small hemisphere fill approximates indirect/global illumination and
    // keeps surfaces readable when the key light is behind them.
    vec3 groundFill = vec3(0.18, 0.17, 0.16);
    vec3 skyFill = vec3(0.30, 0.34, 0.40);
    float hemisphere = normal.y * 0.5 + 0.5;
    vec3 ambient = mix(groundFill, skyFill, hemisphere);
    ambient = max(ambient, vec3(0.22));

    vec3 baseColor = colDiffuse.rgb * (ambient + vec3(diffuse * 0.72));
    vec3 highlight = vec3(1.0, 0.98, 0.94) * (specular * 0.55 + rim * 0.08);
    finalColor = vec4(baseColor + highlight, colDiffuse.a);
}
