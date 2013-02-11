/*
 * GLShader.h
 *
 *  Created on: Jan 23, 2012
 *      Author: worker
 */

#ifndef GLSHADER_H_
#define GLSHADER_H_
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <GL/gl.h>
#include <string>
#include "yuri/log/Log.h"
using namespace yuri::log;

using std::string;

namespace yuri {

namespace graphics {

class GLShader {
public:
	GLShader(Log &log_,GLenum type);
	virtual ~GLShader();
	bool load_file(string filename);
	bool load(string text);
	bool compile();
	GLuint get_shader();
protected:
	Log log;
	GLenum type;
	GLchar *shader_text;
	GLint shader_size;
	GLuint shader_object;
};

}

}

#endif /* GLSHADER_H_ */
