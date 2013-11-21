/*!
 * @file 		GLShader.cpp
 * @author 		Zdenek Travnicek
 * @date 		23.1.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2012 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "GLShader.h"
#include <fstream>
#include <iostream>
#include <GL/glext.h>
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ios;

namespace yuri {

namespace graphics {

GLShader::GLShader(log::Log &log_, GLenum type):log(log_),type(type),shader_text(0),
		shader_size(0),shader_object(0)
{
	log.set_label("[GLShader] ");
}

GLShader::~GLShader()
{
}


bool GLShader::load_file(std::string filename)
{
	ifstream file;
	file.open(filename.c_str(),ios::binary);
	if (file.bad()) return false;
	file.seekg(0,ios::end);
	shader_size = file.tellg();
	file.seekg(0,ios::beg);
	if (!shader_size) return false;
	if (shader_text) delete [] shader_text;
	shader_text = new GLchar[shader_size];
	file.read(shader_text, shader_size);
	return true;
}
bool GLShader::load(std::string text)
{
	shader_size = text.size();
	if (!shader_size) return false;
	if (shader_text) delete [] shader_text;
	shader_text = new GLchar[++shader_size+1];
	std::fill(shader_text, shader_text+shader_size,0);
//	memset(shader_text,0,shader_size+1);
	std::copy_n(text.c_str(), shader_size-1, shader_text);
//	memcpy(shader_text,text.c_str(),shader_size-1);
	//std::cerr << "Allocating " << shader_size << endl;
	shader_text[shader_size] = 0;
	return true;
}

bool GLShader::compile()
{
	shader_object = glCreateShader(type);
	glShaderSource(shader_object,1,(const GLchar**)&shader_text,&shader_size);
	glCompileShader(shader_object);
	GLint compiled;
	glGetObjectParameterivARB(shader_object, GL_COMPILE_STATUS, &compiled);
	if (compiled) {
		log[log::debug] << "Shader " << shader_object << " compiled without problems." << endl;
		return true;
	}
	GLint blen = 0;
	GLsizei slen = 0;
	glGetShaderiv(shader_object, GL_INFO_LOG_LENGTH , &blen);
	if (blen > 1)
	{
	 GLchar* compiler_log = new GLchar[blen];
	 glGetInfoLogARB(shader_object, blen, &slen, compiler_log);
	 log[log::error] << "compiler_log:" <<  compiler_log <<endl;
	 delete compiler_log;
	}
	return false;
}

GLuint GLShader::get_shader()
{
	return shader_object;
}


}

}
