/*!
 * @file 		GLShader.h
 * @author 		Zdenek Travnicek
 * @date 		23.1.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2012 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef GLSHADER_H_
#define GLSHADER_H_
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <GL/gl.h>
#include <string>
#include "yuri/log/Log.h"

namespace yuri {

namespace graphics {

class GLShader {
public:
	GLShader(log::Log &log_,GLenum type);
	virtual ~GLShader();
	bool load_file(std::string filename);
	bool load(std::string text);
	bool compile();
	GLuint get_shader();
protected:
	log::Log log;
	GLenum type;
	GLchar *shader_text;
	GLint shader_size;
	GLuint shader_object;
};

}

}

#endif /* GLSHADER_H_ */
