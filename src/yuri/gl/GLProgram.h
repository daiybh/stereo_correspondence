/*!
 * @file 		GLProgram.cpp
 * @author 		Zdenek Travnicek
 * @date 		23.1.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2012 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef GLPROGRAM_H_
#define GLPROGRAM_H_
#include "GLShader.h"
namespace yuri {

namespace gl {

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
private:
	log::Log log;
	GLuint program;

};

}

}

#endif /* GLPROGRAM_H_ */
