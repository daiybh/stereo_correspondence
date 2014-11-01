/*
 * default_shaders.h
 *
 *  Created on: 31. 10. 2014
 *      Author: neneko
 */

#ifndef DEFAULT_SHADERS_H_
#define DEFAULT_SHADERS_H_

#include <string>

namespace yuri {

namespace gl {

namespace shaders {
namespace {
const std::string vs_default = R"XXX(
void main()
{
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	gl_TexCoord[0] = gl_MultiTexCoord0;
}
)XXX";

const std::string fs_head = R"XXX(
uniform float dx;
uniform float dy;
uniform float tx;
uniform float ty;
uniform int flip_x;
uniform int flip_y;

)XXX";

const std::string fs_main = R"XXX(
void main()
{
	vec2 mapped_coords = transform_coords(gl_TexCoord[0].st * 2.0f - vec2(1.0f, 1.0f));
	mapped_coords = mapped_coords * vec2(1.0f -2.0f * flip_x, 1.0f - 2.0f * flip_y); 	 
	vec2 coord = (mapped_coords*0.5f+vec2(0.5f, 0.5f)) * vec2(tx, ty);
	vec4 color = get_color(coord);
	gl_FragColor = map_color(color, coord);
}
)XXX";

const std::string fs_default_transform = R"XXX(
vec2 transform_coords(vec2 coord) {
	return coord;
}
)XXX";

const std::string fs_default_map = R"XXX(
vec4 map_color(vec4 color, vec2 coord) {
	return color;
}
)XXX";

std::string prepare_fs(const std::string& get_color, const std::string& transform, const std::string& color_map)
{

	return 	fs_head +
			get_color +
			(transform.empty()?fs_default_transform:transform) +
			"\n" +
			(color_map.empty()?fs_default_map:color_map) +
			"\n" +
			fs_main;

}


const std::string fs_get_rgb = R"XXX(
uniform sampler2D tex0;
vec4 get_color(vec2 coord) {
	return texture2D(tex0, coord);
}
)XXX";

const std::string fs_get_yuv444 = R"XXX(
uniform sampler2D tex0;
vec4 get_color(vec2 coord) {
	vec4 col0 = texture2D(tex0, coord);
	float y = col0.r - 0.0625, u = col0.g - 0.5, v = col0.b - 0.5;
	float r = 1.164 * y + 1.596*v, g = 1.164 * y - 0.392* u - 0.813 * v, b = 1.164*y + 2.017 * u;
	return vec4(r, g, b, 1.0);
}
)XXX";


const std::string fs_get_yuyv422 = R"XXX(
uniform sampler2D tex0, tex1;
vec4 get_color(vec2 coord) {
	vec4 col0 = texture2D(tex0, coord);
	vec4 col1 = texture2D(tex1, coord);
	float y = col1.r - 0.0625, u = col0.g - 0.5, v = col0.a - 0.5;
	float r = 1.164 * y + 1.596*v, g = 1.164 * y - 0.392* u - 0.813 * v, b = 1.164*y + 2.017 * u;
	return vec4(r, g, b, 1.0);
}
)XXX";

const std::string fs_get_yvyu422 = R"XXX(
uniform sampler2D tex0, tex1;
vec4 get_color(vec2 coord) {
	vec4 col0 = texture2D(tex0, coord);
	vec4 col1 = texture2D(tex1, coord);
	float y = col1.r - 0.0625, u = col0.a - 0.5, v = col0.g - 0.5;
	float r = 1.164 * y + 1.596*v, g = 1.164 * y - 0.392* u - 0.813 * v, b = 1.164*y + 2.017 * u;
	return vec4(r, g, b, 1.0);
}
)XXX";

const std::string fs_get_uyvy422 = R"XXX(
uniform sampler2D tex0, tex1;
vec4 get_color(vec2 coord) {
	vec4 col0 = texture2D(tex0, coord);
	vec4 col1 = texture2D(tex1, coord);
	float y = col1.a - 0.0625, u = col0.r - 0.5, v = col0.b - 0.5;
	float r = 1.164 * y + 1.596*v, g = 1.164 * y - 0.392* u - 0.813 * v, b = 1.164*y + 2.017 * u;
	return vec4(r, g, b, 1.0);
}
)XXX";

const std::string fs_get_vyuy422 = R"XXX(
uniform sampler2D tex0, tex1;
vec4 get_color(vec2 coord) {
	vec4 col0 = texture2D(tex0, coord);
	vec4 col1 = texture2D(tex1, coord);
	float y = col1.a - 0.0625, u = col0.b - 0.5, v = col0.r - 0.5;
	float r = 1.164 * y + 1.596*v, g = 1.164 * y - 0.392* u - 0.813 * v, b = 1.164*y + 2.017 * u;
	return vec4(r, g, b, 1.0);
}
)XXX";

const std::string fs_get_yuv_planar = R"XXX(
uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;
vec4 get_color(vec2 coord) {
	vec4 col = vec4(
		texture2D(tex0, coord).r,
		texture2D(tex1, coord).r-0.5,
		texture2D(tex2, coord).r-0.5,
		1.0);
	mat4 y2rt = mat4(1, 0, 1.371, 0,
	                 1, -.337, -0.698, 0,
	                 1, 1.733,0, 0,
	                 0, 0, 0, 1);
	return col*y2rt;
}
)XXX";

}
}
}
}



#endif /* DEFAULT_SHADERS_H_ */
