/*!
 * @file 		GL.h
 * @author 		Zdenek Travnicek
 * @date 		31.5.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef GL_H_
#define GL_H_
#include <map>
#include "GLProgram.h"
#include <GL/gl.h>
#include "yuri/io/BasicFrame.h"
#include "yuri/graphics/WindowBase.h"

namespace yuri {

namespace graphics {
using namespace yuri::io;

struct _texture_info {
	GLuint tid[8];
	GLdouble tx, ty, dx, dy;
	bool flip_x, flip_y;
	bool keep_aspect;
	shared_ptr<GLProgram> shader;

	yuri::format_t format;
	GLint texture_units[8];
	yuri::size_t wh;
	_texture_info():tx(0.0f),ty(0.0f),dx(0.0), dy(0.0), flip_x(false),
			flip_y(false),keep_aspect(true),format(YURI_FMT_RGB24),wh(0) {
		for (int i=0;i<8;++i) {
			tid[i]=(GLuint)-1;
			texture_units[i]=-1;
		}
	}
	void load_texture_units(){
		assert(shader);
		char n[]="texX\0x00";
		for (int i=0;i<8;++i) {
			n[3]='0'+i;
			std::string name = std::string(n);
			texture_units[i]=shader->get_uniform(name);
		}
	}
	void bind_texture_units() {
		assert(shader);
		for (int i=0;i<8;++i) {
			if (texture_units[i]<0) continue;
			glActiveTexture(GL_TEXTURE0+i);
			glBindTexture(GL_TEXTURE_2D,tid[i]);
			//std::cerr << "setting uniform " << texture_units[i] << " to " << i << endl;
			shader->set_uniform_sampler(texture_units[i],i);
		}
	}
	inline void gen_texture(int id) {
		if (tid[id]==(GLuint)-1) glGenTextures(1,tid+id);
	}
	void set_tex_coords(double  *v) {
		glTexCoord2dv(v);
		for (int i=0;i<8;++i) {
			if (tid[i]!=(GLuint)-1) {
				glMultiTexCoord2dv(GL_TEXTURE0+i,v);
			}
		}
	}
	void finish_update(Log &log,yuri::format_t fmt,std::string vs,std::string fs){
		format = fmt;
		if (!shader) {
			shader.reset(new GLProgram(log));
			shader->load_shader(GL_VERTEX_SHADER,vs);
			shader->load_shader(GL_FRAGMENT_SHADER,fs);
			shader->link();
			load_texture_units();
		}
	}
};

class GL {
public:
	GL(Log &log_);
	virtual ~GL();
	std::map<uint,_texture_info> textures;
	void generate_texture(uint tid, pBasicFrame frame);
	void generate_empty_texture(yuri::uint_t tid, yuri::format_t fmt, yuri::size_t w, yuri::size_t h);
	void setup_ortho(GLdouble left=0.0, GLdouble right=1.0f,
			GLdouble bottom=0.0, GLdouble top=1.0,
			GLdouble near=-100.0, GLdouble far=100.0);
	void draw_texture(uint tid, shared_ptr<WindowBase> win, GLdouble width=1.0,
			GLdouble height=1.0, GLdouble x=0.0, GLdouble y=0.0);
	static void enable_smoothing();
	static void save_state();
	static void restore_state();
	void enable_depth();
	void set_lq422(yuri::uint_t q);
	bool prepare_texture(yuri::uint_t tid, yuri::uint_t texid, yuri::ubyte_t *data,
			yuri::size_t w, yuri::size_t h, GLenum tex_mode, GLenum data_mode, bool update);
	bool finish_frame();
	Log log;
	static mutex big_gpu_lock;
protected:
	static std::string simple_vertex_shader, simple_fragment_shader,
		fragment_shader_yuv444,
		fragment_shader_yuv422_lq, fragment_shader_yuv422_very_lq,
		fragment_shader_yuv_planar;
	yuri::uint_t lq_422;
};

}

}

#endif /* GL_H_ */
