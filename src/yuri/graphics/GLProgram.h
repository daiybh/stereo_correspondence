/*!
 * @file 		GLProgram.cpp
 * @author 		Zdenek Travnicek
 * @date 		23.1.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2012 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef GLPROGRAM_H_
#define GLPROGRAM_H_
#include "GLShader.h"
namespace yuri {

namespace graphics {

class GLProgram {
public:
	GLProgram(log::Log &log_);
	virtual ~GLProgram();
	bool attach_shader(GLShader &shader);
	bool load_shader(GLuint type,std::string source);
	bool load_shader_file(GLuint type,std::string source);
	bool link();
	void use();
	void stop();
	void bind_attrib(GLuint index, std::string name);
	GLint get_uniform(std::string name);
	void set_uniform_sampler(GLint id, GLint value);
protected:
	log::Log log;
	GLuint program;

};

}

}

#endif /* GLPROGRAM_H_ */
