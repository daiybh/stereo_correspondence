/*!
 * @file 		GL.h
 * @author 		Zdenek Travnicek
 * @date 		31.5.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef GL_H_
#define GL_H_
#include <map>
#include <vector>
#include "GLProgram.h"
#include <GL/gl.h>
#include "yuri/core/forward.h"
#include "yuri/core/frame/raw_frame_types.h"


namespace yuri {

namespace gl {
//typedef yuri::shared_ptr<class WindowBase> pWindowBase;

struct texture_info_t {
	GLuint tid[8];
	GLdouble tx, ty, dx, dy;
	bool flip_x, flip_y;
	bool keep_aspect;
	shared_ptr<GLProgram> shader;

	format_t format;
	GLint texture_units[8];
	GLint uniform_tx, uniform_ty, uniform_dx, uniform_dy, uniform_flip_x, uniform_flip_y;
	size_t wh;
	texture_info_t():tx(0.0f),ty(0.0f),dx(0.0), dy(0.0), flip_x(false),
			flip_y(false),keep_aspect(false),format(0),wh(0) {
		for (int i=0;i<8;++i) {
			tid[i]=static_cast<GLuint>(-1);
			texture_units[i]=-1;
		}
	}
	void load_texture_units(){
		char n[]="texX\0x00";
		for (int i=0;i<8;++i) {
			n[3]='0'+i;
			std::string name = std::string(n);
			texture_units[i]=shader->get_uniform(name);
		}
		uniform_dx = shader->get_uniform("dx");
		uniform_dy = shader->get_uniform("dy");
		uniform_tx = shader->get_uniform("tx");
		uniform_ty = shader->get_uniform("ty");
		uniform_flip_x = shader->get_uniform("flip_x");
		uniform_flip_y = shader->get_uniform("flip_y");
	}
	void bind_texture_units() {
		for (int i=0;i<8;++i) {
			if (texture_units[i]<0) continue;
			glActiveTexture(GL_TEXTURE0+i);
			glBindTexture(GL_TEXTURE_2D,tid[i]);
			//std::cerr << "setting uniform " << texture_units[i] << " to " << i << endl;
			shader->set_uniform_sampler(texture_units[i],i);
		}
		shader->set_uniform_float(uniform_dx, dx);
		shader->set_uniform_float(uniform_dy, dy);
		shader->set_uniform_float(uniform_tx, tx);
		shader->set_uniform_float(uniform_ty, ty);
		shader->set_uniform_int(uniform_flip_x, flip_x);
		shader->set_uniform_int(uniform_flip_y, flip_y);
	}

	inline void gen_texture(int id) {
		if (tid[id]==static_cast<GLuint>(-1)) glGenTextures(1,tid+id);
	}

	void set_tex_coords(double  *v) {
		glTexCoord2dv(v);
		for (int i=0;i<8;++i) {
			if (tid[i]!=static_cast<GLuint>(-1)) {
				glMultiTexCoord2dv(GL_TEXTURE0+i,v);
			}
		}
	}

	bool shader_update_needed(const format_t fmt) const {
		return !shader || fmt != format;
	}

	void finish_update(log::Log &log,yuri::format_t fmt,const std::string& vs, const std::string& fs)
	{
		if (format != fmt) {
			format = fmt;
			shader.reset();
		}
		if (!shader) {
			log[log::debug] << "Loading fs:\n"<<fs << "\nAnd vs:\n"<<vs;
			shader = std::make_shared<GLProgram>(log);
			shader->load_shader(GL_VERTEX_SHADER,vs);
			shader->load_shader(GL_FRAGMENT_SHADER,fs);
			shader->link();
			load_texture_units();
		}
	}
};

class GL {
public:
	GL(log::Log &log_);
	virtual ~GL() noexcept;
	std::map<uint,texture_info_t> textures;
	void generate_texture(index_t tid, const core::pFrame& frame);
	void generate_empty_texture(index_t tid, format_t fmt, resolution_t resolution);
	void setup_ortho(GLdouble left=0.0, GLdouble right=1.0f,
			GLdouble bottom=0.0, GLdouble top=1.0,
			GLdouble near=-100.0, GLdouble far=100.0);
	void draw_texture(index_t tid, resolution_t res = {0, 0}, GLdouble width=1.0,
			GLdouble height=1.0, GLdouble x=0.0, GLdouble y=0.0);
	static void enable_smoothing();
	static void save_state();
	static void restore_state();
	void enable_depth();
	bool prepare_texture(index_t tid, unsigned texid, uint8_t *data,
			resolution_t resolution, GLenum tex_mode, GLenum data_mode, bool update,
			GLenum data_type = GL_UNSIGNED_BYTE);
	bool finish_frame();
	log::Log log;

	std::string transform_shader;
	std::string color_map_shader;
	static mutex big_gpu_lock;
	static std::vector<format_t> get_supported_formats();
};

}

}

#endif /* GL_H_ */
