#version 460
layout(location = 0) out vec4 frag_color;
uniform vec2 u_resolution;
uniform vec3 u_camera_position;
uniform vec3 u_camera_target;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform float u_focal_length;
uniform float u_max_ray_distance;
uniform vec3 u_background_color;
uniform int u_show_grid;
uniform float u_grid_spacing;
uniform int u_grid_plane;
uniform int u_fb_y_up;
vec3 gridA(vec3 ro, vec3 rd, float mt, vec3 col, float s) {
    if(u_show_grid==0) return col;
    vec3 n=u_grid_plane==1?vec3(0.,1.,0.):
             (u_grid_plane==2?vec3(1.,0.,0.):vec3(0.,0.,1.));
    float den=dot(rd,n);
    if(abs(den)<1e-6) return col;
    float tt=-dot(ro,n)/den;
    if(tt<=0. || tt>=mt) return col;
    vec3 p=ro+rd*tt;
    vec2 g=u_grid_plane==1?p.xz:(u_grid_plane==2?p.yz:p.xy);
    vec2 w=fwidth(g);
    vec2 a=abs(fract(g/u_grid_spacing+.5)-.5)*u_grid_spacing;
    float line=1.-smoothstep(0.,max(max(w.x,w.y),1e-5)*1.5,min(a.x,a.y));
    // Fade over a distance proportional to the cell size (1 m baseline), so
    // coarse grids (km work) stay visible across their own cells.
    float ft=tt/max(u_grid_spacing,1.);
    float fade=clamp(1./(1.+ft*ft*.002),0.,1.);
    return mix(col,vec3(.62,.75,.92),line*s*fade);
}
// One world axis (line through the origin along `axis`) drawn with a
// screen-space-constant width via ray-vs-line distance, like a normal CAD.
vec3 axisLine(vec3 ro, vec3 rd, float mt, vec3 col, vec3 axis, vec3 acol) {
    if(u_show_grid==0) return col;
    float b=dot(rd,axis);
    float den=1.-b*b;
    if(abs(den)<1e-6) return col;          // ray parallel to the axis
    float d=dot(rd,ro);
    float e=dot(axis,ro);
    float t=(b*e-d)/den;                    // closest param along the camera ray
    if(t<=0.||t>=mt) return col;
    float s=(e-b*d)/den;                    // closest param along the axis
    vec3 pr=ro+rd*t;
    vec3 pa=axis*s;
    float dist=length(pr-pa);
    float wpp=t*2./(u_focal_length*max(u_resolution.y,1.));  // world units / pixel
    float px=dist/max(wpp,1e-9);
    float linev=1.-smoothstep(.9,2.2,px);
    float ft=t/max(u_grid_spacing,1.);
    float fade=clamp(1./(1.+ft*ft*.0008),0.,1.);
    return mix(col,acol,linev*fade);
}
void main() {
    vec2 px = gl_FragCoord.xy;
    vec2 uv = (px - 0.5*u_resolution)/max(u_resolution.y, 1.0);
    if (u_fb_y_up == 0) uv.y = -uv.y;
    vec3 fwd = normalize(u_camera_target - u_camera_position);
    vec3 rd = normalize(2.0*uv.x*normalize(u_camera_right)
                      + 2.0*uv.y*normalize(u_camera_up) + u_focal_length*fwd);
    vec3 col = gridA(u_camera_position, rd, u_max_ray_distance,
                     u_background_color, 0.6);
    col = axisLine(u_camera_position, rd, u_max_ray_distance, col,
                   vec3(1.,0.,0.), vec3(1.00,0.34,0.25));   // X red
    col = axisLine(u_camera_position, rd, u_max_ray_distance, col,
                   vec3(0.,1.,0.), vec3(0.33,0.92,0.41));   // Y green
    col = axisLine(u_camera_position, rd, u_max_ray_distance, col,
                   vec3(0.,0.,1.), vec3(0.36,0.57,1.00));   // Z blue
    frag_color = vec4(col, 1.0);
}
