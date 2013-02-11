/*
 * GLProgram.cpp
 *
 *  Created on: Jan 23, 2012
 *      Author: worker
 */

#include "GLProgram.h"

namespace yuri {

namespace graphics {

GLProgram::GLProgram(Log &log_):log(log_)
{
	log.setLabel("[GLProgram] ");
	program = glCreateProgram();
}


GLProgram::~GLProgram()
{

}

bool GLProgram::attach_shader(GLShader &shader)
{
	glAttachShader(program,shader.get_shader());
	GLenum err = glGetError();
	if (err) {
		log[error] << "Error " << err << " while attaching shader" << endl;
	}
	return true;
}

bool GLProgram::link()
{
	log[debug] << "Linking program" << endl;
	glLinkProgram(program);
	GLint linked;
	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	if (linked) {
		/*view_matrix = glGetUniformLocation(program,"view_matrix");
		model_matrix = glGetUniformLocation(program,"model_matrix");
		projection_matrix  = glGetUniformLocation(program,"projection_matrix");
		cerr << "vm: " << view_matrix << ", mm: " << model_matrix << ", pm: " << projection_matrix << endl;*/
		log[debug] << "Shader program linked" <<endl;
		return true;
	}
	log[warning] << "Shader program NOT linked" <<endl;
	GLint blen = 0;
	GLsizei slen = 0;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH , &blen);
	if (blen > 1)
	{
		GLchar* compiler_log = new GLchar[blen];
		glGetInfoLogARB(program, blen, &slen, compiler_log);
		log[error] << "compiler_log:" <<  compiler_log <<endl;
		delete compiler_log;
	}
	return false;
}

void GLProgram::use()
{
	glUseProgram(program);
}
void GLProgram::stop()
{
	glUseProgram(0);
}

bool GLProgram::load_shader_file(GLuint type, string source)
{
	GLShader shader(log,type);
	log[debug]<<"Loading shader" <<endl;
	if (!shader.load_file(source)) {
	log[error] << "Failed to load shader " << source << endl;
		return false;
	}
	log[verbose_debug]<<"Compiling " << source << endl;
	if (!shader.compile()) {
	    log[error] << "Failed to compile shader "<< source << endl;
	    return false;
	}
	return attach_shader(shader);
}

bool GLProgram::load_shader(GLuint type, string source)
{
	GLShader shader(log,type);
	log[debug]<<"Loading shader" << endl;
	log[verbose_debug] << source << endl;
	if (!shader.load(source)) {
	log[error] << "Failed to load shader " << source << endl;
		return false;
	}
	log[debug]<<"Compiling shader" << endl;
	log[verbose_debug] << source << endl;
	if (!shader.compile()) {
	    log[error] << "Failed to compile shader "<< source << endl;
	    return false;
	}
	return attach_shader(shader);
}

void GLProgram::bind_attrib(GLuint index,string name)
{
	glBindAttribLocation(program,index,name.c_str());
}
GLint GLProgram::get_uniform(string name)
{
	return glGetUniformLocation(program,const_cast<const char *>(name.c_str()));
}
void GLProgram::set_uniform_sampler(GLint id, GLint value)
{
	return glUniform1i(id,value);
}
}

}
