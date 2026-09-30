#version 330

in vec3 worldPosition;

uniform vec3 viewPos;
uniform vec4 colDiffuse;

out vec4 finalColor;

void main() {
    vec3 normal = normalize(cross(dFdx(worldPosition), dFdy(worldPosition)));
    if (!gl_FrontFacing) {
        normal = -normal;
    }

    vec3 lightDirection = normalize(vec3(-0.45, 0.8, 0.35));
    vec3 viewDirection = normalize(viewPos - worldPosition);
    vec3 halfDirection = normalize(lightDirection + viewDirection);

    float diffuse = max(dot(normal, lightDirection), 0.0);
    float specular = pow(max(dot(normal, halfDirection), 0.0), 64.0);
    float rim = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.0);

    vec3 baseColor = colDiffuse.rgb * (0.16 + diffuse * 0.72);
    vec3 highlight = vec3(1.0, 0.98, 0.94) * (specular * 0.55 + rim * 0.08);
    finalColor = vec4(baseColor + highlight, colDiffuse.a);
}
