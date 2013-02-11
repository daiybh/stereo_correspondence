/*
 * GLProgram.h
 *
 *  Created on: Jan 23, 2012
 *      Author: worker
 */

#ifndef GLPROGRAM_H_
#define GLPROGRAM_H_
#include "GLShader.h"
namespace yuri {

namespace graphics {

class GLProgram {
public:
	GLProgram(Log &log_);
	virtual ~GLProgram();
	bool attach_shader(GLShader &shader);
	bool load_shader(GLuint type, string source);
	bool load_shader_file(GLuint type, string source);
	bool link();
	void use();
	void stop();
	void bind_attrib(GLuint index,string name);
	GLint get_uniform(string name);
	void set_uniform_sampler(GLint id, GLint value);
protected:
	Log log;
	GLuint program;

};

}

}

#endif /* GLPROGRAM_H_ */
