#version 100

attribute vec4 position;  // xyz = position, w = point_size

uniform mat4 vp;
uniform vec3 color;      // RGB only

varying vec3 v_color;

void main() {
    gl_Position = vp * vec4(position.xyz, 1.0);
    gl_PointSize = position.w;  // Point size from vertex attribute
    v_color = color;
}
