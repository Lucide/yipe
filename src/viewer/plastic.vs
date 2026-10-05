#version 330

in vec3 vertexPosition;
in vec3 vertexNormal;

uniform mat4 mvp;
uniform mat4 matModel;

out vec3 worldPosition;
out vec3 worldNormal;

void main() {
    vec4 position = matModel * vec4(vertexPosition, 1.0);
    worldPosition = position.xyz;
    worldNormal = normalize(mat3(matModel) * vertexNormal);
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
