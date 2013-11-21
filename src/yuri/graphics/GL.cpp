/*!
 * @file 		GL.cpp
 * @author 		Zdenek Travnicek
 * @date 		22.10.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "GL.h"
#include "yuri/core/pipe/Pipe.h"
#include "yuri/core/thread/IOThread.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/raw_frame_params.h"
#include <cassert>

namespace yuri {

namespace graphics {
mutex GL::big_gpu_lock;

std::string GL::simple_vertex_shader(
		"void main()\n"
		"{\n"
		"gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
		"gl_TexCoord[0] = gl_MultiTexCoord0;\n"
		"}\n");
std::string GL::fragment_shader_yuv422_very_lq(
		"uniform sampler2D tex0;\n"
		"void main()\n"
		"{\n"
		"vec4 col = texture2D(tex0, gl_TexCoord[0].st);\n"
		"float y = col.r - 0.0625 , u = col.g - 0.5, v = col.a - 0.5;\n"
		"float r = 1.164 * y + 1.596*v, g = 1.164 * y - 0.392* u - 0.813 * v, b = 1.164*y + 2.017 * u;\n"
		"gl_FragColor = vec4(r, g, b, 1.0);\n"
		"}\n");
std::string GL::fragment_shader_uyvy422_very_lq(
		"uniform sampler2D tex0;\n"
		"void main()\n"
		"{\n"
		"vec4 col = texture2D(tex0, gl_TexCoord[0].st);\n"
		"float y = col.g - 0.0625 , u = col.r - 0.5, v = col.b - 0.5;\n"
		"float r = 1.164 * y + 1.596*v, g = 1.164 * y - 0.392* u - 0.813 * v, b = 1.164*y + 2.017 * u;\n"
		"gl_FragColor = vec4(r, g, b, 1.0);\n"
		"}\n");

/*
 *
 *std::string GL::simple_vertex_shader =
	"#version 150\n"
	"in vec3 position;\n"
	"in vec3 normal;\n"
	"in vec2 texCoord;\n"

	"out VertexData {\n"
	"	vec2 texCoord;\n"
	"	vec3 normal;\n"
	"} VertexOut;\n"
	"void main()\n"
	"{\n"
		"VertexOut.texCoord = texCoord;\n"
		//"VertexOut.normal = normalize(normalMatrix * normal);\n"
		"gl_Position = vec4(position, 1.0f);\n"
	"}\n";


string GL::fragment_shader_yuv422_very_lq(
		"#version 150 \n"
		"uniform sampler2D tex0;\n"
		"in VertexData {\n"
		"	vec2 texCoord;\n"
		"	vec3 normal;\n"
		"} vertex;\n"
		"out vec4 color;\n"
		"void main()\n"
		"{\n"
		"vec4 col = texture2D(tex0, vertex.texCoord.st);\n"
		"float y = col.r - 0.0625 , u = col.g - 0.5, v = col.a - 0.5;\n"
		"float r = 1.164 * y + 1.596*v, g = 1.164 * y - 0.392* u - 0.813 * v, b = 1.164*y + 2.017 * u;\n"
		"color = vec4(r, g, b, 1.0);\n"
		"}\n");
 *
 */

std::string GL::simple_fragment_shader(
		"uniform sampler2D tex0;\n"
		"void main()\n"
		"{\n"
		"gl_FragColor = texture2D(tex0, gl_TexCoord[0].st);\n"
		"}\n");

std::string GL::fragment_shader_yuv422_lq(
		"uniform sampler2D tex0, tex1;\n"
		"void main()\n"
		"{\n"
		"vec4 col0 = texture2D(tex0, gl_TexCoord[0].st);\n"
		"vec4 col1 = texture2D(tex1, gl_TexCoord[0].st);\n"
		"float y = col1.r - 0.0625, u = col0.g - 0.5, v = col0.a - 0.5;\n"
		"float r = 1.164 * y + 1.596*v, g = 1.164 * y - 0.392* u - 0.813 * v, b = 1.164*y + 2.017 * u;\n"
		"gl_FragColor = vec4(r, g, b, 1.0);\n"
		"}\n");
std::string GL::fragment_shader_yuv444(
		"uniform sampler2D tex0;\n"
		"void main()\n"
		"{\n"
		"vec4 col0 = texture2D(tex0, gl_TexCoord[0].st);\n"
		"float y = col0.r - 0.0625, u = col0.g - 0.5, v = col0.b - 0.5;\n"
		"float r = 1.164 * y + 1.596*v, g = 1.164 * y - 0.392* u - 0.813 * v, b = 1.164*y + 2.017 * u;\n"
		"gl_FragColor = vec4(r, g, b, 1.0);\n"
		"}\n");
/// @bug: This is actually the LQ version....
std::string GL::fragment_shader_uyvy422_lq(
		"uniform sampler2D tex0, tex1;\n"
		"void main()\n"
		"{\n"
		"vec4 col0 = texture2D(tex0, gl_TexCoord[0].st);\n"
		"vec4 col1 = texture2D(tex1, gl_TexCoord[0].st);\n"
		//"float y = 0.5/*col1.g - 0.0625*/, u = col0.r - 0.5, v = col0.b - 0.5;\n"

		"float y = col0.g - 0.0625 , u = col0.r - 0.5, v = col0.b - 0.5;\n"
		"float r = 1.164 * y + 1.596*v, g = 1.164 * y - 0.392* u - 0.813 * v, b = 1.164*y + 2.017 * u;\n"
		"gl_FragColor = vec4(r, g, b, 1.0);\n"
		"}\n");

std::string GL::fragment_shader_yuv_planar(
		"uniform sampler2D tex0;\n"
		"uniform sampler2D tex1;\n"
		"uniform sampler2D tex2;\n"
		"void main()\n"
		"{\n"
		"vec4 col = vec4(\n"
		"	texture2D(tex0, gl_TexCoord[0].st).r,\n"
		"	texture2D(tex1, gl_TexCoord[0].st).r-0.5,\n"
		"	texture2D(tex2, gl_TexCoord[0].st).r-0.5,\n"
		"	1.0);\n"
		"mat4 y2rt = mat4(1, 0, 1.371, 0,\n"
		"                 1, -.337, -0.698, 0,\n"
		"                 1, 1.733,0, 0,\n"
		"                 0, 0, 0, 1);\n"
		"gl_FragColor = col*y2rt;\n"
		"}\n");

GL::GL(log::Log &log_):log(log_),lq_422(0)
{
	log.set_label("[GL] ");
}

GL::~GL() {

}

void GL::generate_texture(index_t tid, const core::pFrame& gframe)
{
	using namespace yuri::core;
	core::pRawVideoFrame frame = dynamic_pointer_cast<RawVideoFrame>(gframe);
	if (!frame) return;
//	assert(frame);

	GLdouble &tx = textures[tid].tx;
	GLdouble &ty = textures[tid].ty;

	if (textures[tid].tid[0]==(GLuint)-1) {
		textures[tid].gen_texture(0);
		log[log::info] << "Generated texture " << textures[tid].tid[0] <<"\n";
	}

	GLuint &tex = textures[tid].tid[0];

	glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);
	glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_FALSE);
	glPixelStorei(GL_UNPACK_LSB_FIRST, GL_FALSE);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
	glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
	glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_SWAP_BYTES, GL_FALSE);
	glPixelStorei(GL_PACK_LSB_FIRST, GL_FALSE);
	glPixelStorei(GL_PACK_ROW_LENGTH, 0);
	glPixelStorei(GL_PACK_SKIP_ROWS, 0);
	glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	resolution_t res = frame->get_resolution();
	yuri::size_t w = res.width;
	yuri::size_t h = res.height;
	yuri::size_t wh = w > h ? w : h;
	for (yuri::size_t j = 128; j < 9000; j *= 2) {
		if (wh <= j) {
			wh = j;
			break;
		}
	}
//	log[log::info] << "w: " << w << ", h: " <<h<< ", wh: " << wh << "\n";
	tx = (float) w / (float) wh;
	ty = (float) h / (float) wh;
	glBindTexture(GL_TEXTURE_2D, tex);
	glEnable(GL_MULTISAMPLE);
	glSampleCoverage(0.1,GL_TRUE);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//log[log::info] << "wh: " << wh << ", tx: " << tx << ", format: " << frame->get_format() <<"\n";
//	FormatInfo_t fi = core::BasicPipe::get_format_info(frame->get_format());
	const raw_format::raw_format_t& fi = raw_format::get_format_info(frame->get_format());
	size_t plane_count = fi.planes.size();
	switch (frame->get_format()) {
		case raw_format::rgb24:
		case raw_format::rgba32: {
			assert(plane_count==1);
			size_t bpp = fi.planes[0].bit_depth.first/fi.planes[0].bit_depth.second;
			GLenum fmt;
			switch (bpp) {
			case 24: fmt=GL_RGB;break;
			case 32: fmt=GL_RGBA;break;
			default:log[log::warning] <<"Bad input frame"<<"\n";fmt=GL_RGB;break;
			}
			if (wh != textures[tid].wh) {
				uint8_t *image = new uint8_t[(wh * wh * bpp) >> 3];
				prepare_texture(tid,0,image, {wh, wh},fmt,fmt,false);
				delete[] image;
				textures[tid].wh = wh;
			}
			prepare_texture(tid,0,PLANE_RAW_DATA(frame,0), {w, h},fmt,fmt,true);
			textures[tid].finish_update(log,frame->get_format(),simple_vertex_shader,simple_fragment_shader);
		}break;
		case raw_format::bgr24:{
			assert(plane_count==1);
			if (wh != textures[tid].wh) {
				uint8_t *image = new uint8_t[wh * wh * 3];
				prepare_texture(tid,0,image, {wh, wh}, GL_RGB,GL_BGR,false);
				delete[] image;
				textures[tid].wh = wh;
			}
			prepare_texture(tid,0,PLANE_RAW_DATA(frame,0), {w, h} ,GL_RGB,GL_BGR,true);
			textures[tid].finish_update(log,frame->get_format(),simple_vertex_shader,simple_fragment_shader);
		}break;
		case raw_format::abgr32:{
			assert(plane_count==1);
			if (wh != textures[tid].wh) {
				uint8_t *image = new uint8_t[wh * wh * 4];
				prepare_texture(tid,0,image, {wh, wh},GL_RGBA,GL_BGRA,false);
				delete[] image;
				textures[tid].wh = wh;
			}
			prepare_texture(tid,0,PLANE_RAW_DATA(frame,0), {w, h},GL_RGBA,GL_BGRA,true);
			textures[tid].finish_update(log,frame->get_format(),simple_vertex_shader,simple_fragment_shader);
		}break;
		case raw_format::yuv444:
		case raw_format::yuyv422:
		case raw_format::uyvy422:
		{
			if (wh != textures[tid].wh) {
				uint8_t *image;
				if (!lq_422 || frame->get_format()==raw_format::yuv444) {
					image = new uint8_t[wh * wh * 3];
					prepare_texture(tid,0,image, {wh, wh} ,3,GL_RGB,false);
				} else {
					image = new uint8_t[wh * wh * 2];
					prepare_texture(tid,0,image,{wh/2, wh},4,GL_RGBA,false);
					if (lq_422==1) {
						prepare_texture(tid,1,image,{wh, wh},GL_LUMINANCE8_ALPHA8,GL_LUMINANCE_ALPHA,false);
					}
				}
				delete[] image;
				textures[tid].wh = wh;
			}
			if (frame->get_format()==raw_format::yuv444) {
				prepare_texture(tid,0,PLANE_RAW_DATA(frame,0), {w, h},3,GL_RGB,true);
			} else if (!lq_422 ) {
				uint8_t *img = new uint8_t [w*h*3];
				uint8_t *p = PLANE_RAW_DATA(frame,0);
				uint8_t *ty = img, *tu=img+1, *tv=img+2;
//				uint8_t y,u,v;
				if (frame->get_format()==raw_format::yuyv422) {
					for (int i=0;i<static_cast<int>(w*h/2);++i) {
						*ty = *p++;
						ty+=3;
						*tu = *p;
						tu+=3;
						*tu = *p++;
						tu+=3;
						*ty = *p++;
						ty+=3;
						*tv = *p;
						tv+=3;
						*tv = *p++;
						tv+=3;
					}
				} else {
					for (int i=0;i<static_cast<int>(w*h/2);++i) {
						*tu = *p;
						tu+=3;
						*tu = *p++;
						tu+=3;
						*ty = *p++;
						ty+=3;
						*tv = *p;
						tv+=3;
						*tv = *p++;
						tv+=3;
						*ty = *p++;
						ty+=3;

					}
				}
				prepare_texture(tid, 0, img, {w, h}, 3, GL_RGB, true);
				delete [] img;
			} else {
				prepare_texture(tid, 0, PLANE_RAW_DATA(frame,0), {w/2, h}, 4, GL_RGBA, true);
				if (lq_422==1) {
					prepare_texture(tid, 1, PLANE_RAW_DATA(frame,0), {w, h}, GL_LUMINANCE8_ALPHA8, GL_LUMINANCE_ALPHA, true);
				}
			}
			std::string fs;
//			if (frame->get_format()==YURI_FMT_UYVY422) {
//				if (!lq_422) fs = fragment_shader_uyvy444;
//				else if (lq_422==1) fs = fragment_shader_uyvy422_lq;
//				else fs = fragment_shader_uyvy422_very_lq;
//			} else {
				if (!lq_422 || frame->get_format()==raw_format::yuv444) fs = fragment_shader_yuv444;
				else if (lq_422==1) fs = frame->get_format()==raw_format::yuyv422?fragment_shader_yuv422_lq:fragment_shader_uyvy422_lq;
				else fs = frame->get_format()==raw_format::uyvy422?fragment_shader_yuv422_very_lq:fragment_shader_uyvy422_very_lq;
//			}
			textures[tid].finish_update(log,frame->get_format(),simple_vertex_shader,fs);
		}break;
/*

//		case YURI_FMT_YUV420_PLANAR:
		case raw_format::yuv422p:
		case raw_format::yuv444p:{
			assert(fi && fi->planes>=3);
			if (wh != textures[tid].wh) {
				uint8_t *image;
				image = new uint8_t[wh * wh];
				for (int i=0;i<3;++i) {
					prepare_texture(tid,i,image,wh/fi->plane_x_subs[i], wh/fi->plane_y_subs[i],GL_LUMINANCE8,GL_LUMINANCE,false);
				}
				delete[] image;
				textures[tid].wh = wh;
			}
			for (int i=0;i<3;++i) {
				prepare_texture(tid,i,PLANE_RAW_DATA(frame,i),w/fi->plane_x_subs[i],
						h/fi->plane_y_subs[i],GL_LUMINANCE8,GL_LUMINANCE,true);
			}
			textures[tid].finish_update(log,frame->get_format(),simple_vertex_shader,
					fragment_shader_yuv_planar);

		}break;
		case raw_format::r8:
		case raw_format::g8:
		case raw_format::b8:
		case raw_format::y8:
		case raw_format::u8:
		case raw_format::v8:
		case raw_format::depth8: {
			assert(fi && fi->planes==1);
			if (wh != textures[tid].wh) {
				uint8_t *image;
				image = new uint8_t[wh * wh];
				prepare_texture(tid,0,image,wh, wh,GL_LUMINANCE8,GL_LUMINANCE,false);
				delete[] image;
				textures[tid].wh = wh;
			}
			prepare_texture(tid,0,PLANE_RAW_DATA(frame,0),w,	h,GL_LUMINANCE8,GL_LUMINANCE,true);
			textures[tid].finish_update(log,frame->get_format(),simple_vertex_shader,
					simple_fragment_shader);

		}break;
		case raw_format::r16:
		case raw_format::g16:
		case raw_format::b16:
		case raw_format::y16:
		case raw_format::u16:
		case raw_format::v16:
		case raw_format::depth16: {
			assert(fi && fi->planes==1);
			if (wh != textures[tid].wh) {
				uint8_t *image;
				image = new uint8_t[wh * wh*2];
				prepare_texture(tid,0,image,wh, wh,GL_LUMINANCE16,GL_LUMINANCE,false,GL_UNSIGNED_SHORT);
				delete[] image;
				textures[tid].wh = wh;
			}
			prepare_texture(tid,0,PLANE_RAW_DATA(frame,0),w,	h,GL_LUMINANCE16,GL_LUMINANCE,true,GL_UNSIGNED_SHORT);
			textures[tid].finish_update(log,frame->get_format(),simple_vertex_shader,
					simple_fragment_shader);

		}break;
		*/
/*
		case YURI_FMT_DXT1:
		case YURI_FMT_DXT1_WITH_MIPMAPS:
//		case YURI_FMT_DXT2:
//		case YURI_FMT_DXT2_WITH_MIPMAPS:
		case YURI_FMT_DXT3:
		case YURI_FMT_DXT3_WITH_MIPMAPS:
//		case YURI_FMT_DXT4:
//		case YURI_FMT_DXT4_WITH_MIPMAPS:
		case YURI_FMT_DXT5:
		case YURI_FMT_DXT5_WITH_MIPMAPS:{
			GLenum format;
			yuri::format_t yf = frame->get_format();
			yuri::size_t fsize;
			if (yf==YURI_FMT_DXT1 || yf==YURI_FMT_DXT1_WITH_MIPMAPS) {
				format = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
				fsize = wh * wh >>1;
			} else if (yf==YURI_FMT_DXT3 || yf==YURI_FMT_DXT3_WITH_MIPMAPS) {
				format = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
				fsize = wh * wh;
			} else if (yf==YURI_FMT_DXT5 || yf==YURI_FMT_DXT5_WITH_MIPMAPS) {
				format = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
				fsize = wh * wh;
			} else break;
			bool mipmaps = true;
			if ((yf==YURI_FMT_DXT5) || (yf==YURI_FMT_DXT3) || (yf==YURI_FMT_DXT1))
				mipmaps = false;
			if (textures[tid].wh != wh) {
				yuri::size_t remaining=fsize, wh2=wh,next_level = fsize, offset = 0, level =0;
				char *image = new char[fsize];
				while (next_level <= remaining) {
					glCompressedTexImage2D(GL_TEXTURE_2D, level++, format, wh2, wh2, 0, next_level,image);
					if (!mipmaps) break;
					wh2>>=1;
					if (remaining<next_level) break;
					remaining-=next_level;
					offset+=next_level;
					next_level>>=2;
				}

				glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
//				GLenum e = glGetError();
//				log[log::error] << "compressed texture, " << e <<"\n";
				delete[] image;
				textures[tid].wh = wh;
			}
			glBindTexture(GL_TEXTURE_2D, tex);
			fsize=w*h>>((yf==YURI_FMT_DXT1)?1:0);

			yuri::size_t remaining=PLANE_SIZE(frame,0), w2=w, h2=h, next_level = fsize, offset = 0, level =0;
			while (next_level <= remaining) {
				log[log::debug] << "next_level: " << next_level << ", rem: " << remaining <<"\n";
				glCompressedTexSubImage2D(GL_TEXTURE_2D, level++, 0, 0, w2, h2,	format, next_level, PLANE_RAW_DATA(frame,0)+offset);
				if (!mipmaps) break;
				w2>>=1;h2>>=1;
				if (remaining<next_level) break;
				remaining-=next_level;
				offset+=next_level;
				next_level>>=2;
			}
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_BASE_LEVEL, 0);
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL, level-1);
			//GLenum e =
			glGetError();
			textures[tid].finish_update(log,frame->get_format(),simple_vertex_shader,
								simple_fragment_shader);
		} break;

		*/
	} /*else {
		log[log::debug] << "Frame with unsupported format! (" <<
				BasicPipe::get_format_string(frame->get_format()) <<
				")" <<"\n";
	}
	log[log::debug] << "Generated texture " << wh << "x" << wh << " from image " <<
			w << "x" << h << " (" << tx << ", " << ty << ")" <<"\n";
	 */
	glPopClientAttrib();
}
void GL::generate_empty_texture(index_t tid, yuri::format_t fmt, resolution_t resolution)
{
	core::pFrame dummy = core::RawVideoFrame::create_empty(fmt,resolution);
	generate_texture(tid,dummy);
}
void GL::setup_ortho(GLdouble left, GLdouble right,	GLdouble bottom,
		GLdouble top, GLdouble near, GLdouble far)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(left, right, bottom, top, near, far);
	glClearColor(0,0,0,0);
	//glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void GL::draw_texture(index_t tid, shared_ptr<WindowBase> win,  GLdouble width,
		GLdouble height, GLdouble x, GLdouble y)
{
	GLuint &tex = textures[tid].tid[0];
	if (tex==(GLuint)-1) return;
	bool &keep_aspect = textures[tid].keep_aspect;

	glPushAttrib(GL_DEPTH_BUFFER_BIT|GL_ENABLE_BIT|GL_POLYGON_BIT|GL_SCISSOR_BIT|GL_TEXTURE_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_CULL_FACE);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_TEXTURE_2D);
	GLdouble x_0, x_1, y_0, y_1;
	GLdouble &dx = textures[tid].dx;
	GLdouble &dy = textures[tid].dy;
	if (textures[tid].flip_x) {
		x_0 = textures[tid].tx-(dx>0.0?0.0:-dx);
		x_1 = (dx>0.0?dx:0.0);
	} else {
		x_0 = (dx>0.0?dx:0.0);
		x_1 = textures[tid].tx-(dx>0.0?0.0:-dx);
	}
	if (textures[tid].flip_y) {
		y_0 = textures[tid].ty;
		y_1 = (dy>0.0?dy:0.0)-(dy>0.0?0.0:-dy);
	} else {
		y_0 = (dy>0.0?dy:0.0);
		y_1 = textures[tid].ty-(dy>0.0?0.0:-dy);
	}
	glBindTexture(GL_TEXTURE_2D, tex);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	if (textures[tid].shader) {
		textures[tid].shader->use();
		textures[tid].bind_texture_units();
	}
	glActiveTexture(GL_TEXTURE0);
	glBegin(GL_QUADS);
	double tex_coords[][2]={	{x_0, y_1},
							{x_1, y_1},
							{x_1, y_0},
							{x_0, y_0}};
	if (!keep_aspect || !win) {
		textures[tid].set_tex_coords(tex_coords[0]);
		glVertex2f(x, y);
		textures[tid].set_tex_coords(tex_coords[1]);
		glVertex2f(width + x, y);
		textures[tid].set_tex_coords(tex_coords[2]);
		glVertex2f(width + x, height + y);
		textures[tid].set_tex_coords(tex_coords[3]);
		glVertex2f(x, height + y);
	} else {
		float lx,ly;
		float h = win->get_width()*textures[tid].ty/textures[tid].tx;
		if (h < win->get_height()) {
			lx = 0.0f;
			ly= 0.5f - (h/(2.0f*win->get_height()));
		} else {
			ly = 0.0f;
			h = win->get_height()*textures[tid].tx/textures[tid].ty;
			lx = 0.5f - (h/(2.0f*win->get_width()));
		}
//		GLdouble vx_0, vx_1, vy_0, vy_1;
//		vx_0 = x + lx/width;
//		vx_1 = x + width - lx/width;
//		vy_0 = y + ly/height;
//		vy_1 = y + height- lx/height;
		glTexCoord2f(x_0, y_1);
		glVertex2f(lx, ly);
		glTexCoord2f(x_1, y_1);
		glVertex2f(1.0f - lx , ly);
		glTexCoord2f(x_1, y_0);
		glVertex2f(1.0f - lx, 1.0f - ly);
		glTexCoord2f(x_0, y_0);
		glVertex2f(lx, 1.0f -ly);
	}
	glEnd();
	if (textures[tid].shader) textures[tid].shader->stop();
	glBindTexture(GL_TEXTURE_2D,0);
	glPopAttrib();

}
void GL::enable_smoothing()
{
	glShadeModel(GL_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
}

void GL::enable_depth()
{
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
}

void GL::save_state()
{
	GLint matrix;
	glGetIntegerv(GL_MATRIX_MODE,&matrix);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glMatrixMode(matrix);
}
void GL::restore_state()
{
	GLint matrix;
	glGetIntegerv(GL_MATRIX_MODE,&matrix);
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(matrix);
}
/** \brief Sets quality of yuv422 texture rendering
 *
 * When set to 0, 422 image gets upsampled to 444 and then rendered. Highest quality.
 * When set to 1, the image will be rendered using two separate textures,
 * 		using full scale Y component, but leading to degradation in Y and V components.
 * When set to 2, the image will be rendered using single texture, effectively
 * 		equal to downsampling Y. Fastest method.
 *
 */
void GL::set_lq422(int q)
{
	if (q>2) lq_422=2;
	else lq_422 = q;
	log[log::info] << "Rendering in quality " << lq_422 <<"\n";
}
bool GL::prepare_texture(index_t tid, unsigned texid, uint8_t *data,
		resolution_t resolution, GLenum tex_mode, GLenum data_mode, bool update,
		GLenum data_type)
{
	GLenum err;
	glGetError();
	if (!update) textures[tid].gen_texture(texid);
	glBindTexture(GL_TEXTURE_2D, textures[tid].tid[texid]);
	err = glGetError();
	if (err) {
		log[log::error]<< "Error " << err << " while binding texture" <<"\n";
		return false;
	}
	if (!update) {
		glTexImage2D(GL_TEXTURE_2D, 0, tex_mode, resolution.width, resolution.height, 0, data_mode, data_type, 0);
	} else {
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, resolution.width, resolution.height, data_mode, data_type, data);
	}
	err = glGetError();
	if (err) {
		log[log::error] << "Error " << err /*<< ":" << glGetString(err) */<< " uploading tex. data" <<"\n";
		return false;
	}
	if (!update) {
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAX_ANISOTROPY_EXT,32.0);
		err = glGetError();
		if (err) {
			log[log::error] << "Error " << err << " setting texture params" <<"\n";
			return false;
		}
	}
	return true;
}
bool GL::finish_frame()
{
	glFinish();
	return true;
}
}

}
