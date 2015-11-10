/*
 * wrapper.cpp
 *
 *  Created on: Oct 5, 2010
 *      Author: worker
 */

#include "wrapper/WrapperLoader.h"

#include <cstdlib>

//void __attribute__ ((constructor)) __yuri_wrapper_loader(void);
void *get_ptr(const std::string& name);
std::string get_config_name();
std::string get_grabber_name();

#define GET_FUNC_PTR(type,ptr,name) type ptr=reinterpret_cast<type>(reinterpret_cast<uintptr_t>(get_ptr(name)));

using namespace yuri;

namespace {

log::Log& get_logger() {
	static log::Log logger(std::clog);
	logger.set_label("[YURI Wrapper] ");
	return logger;
}
wrapper::WrapperLoader& get_wrap_builder()
{
	static wrapper::WrapperLoader builder(get_logger(), get_config_name(), get_grabber_name());
	return builder;
}

}
extern "C" {
typedef void (*glXSwapBuffers_t)(Display * dpy,	GLXDrawable drawable);

void glXSwapBuffers(Display * dpy,	GLXDrawable drawable)
{
	GET_FUNC_PTR(glXSwapBuffers_t, f, "glXSwapBuffers");
	get_logger()[log::verbose_debug] << "glxSwapBuffers (" << dpy << ", " << drawable <<")";
	get_wrap_builder().pre_swap();
	if (f) f(dpy, drawable);
	get_wrap_builder().post_swap();

}


typedef void (*glViewport_t)(GLint x, GLint y, GLsizei width, GLsizei height);

void glViewport(GLint x, GLint y, GLsizei width, GLsizei height)
{
	GET_FUNC_PTR(glViewport_t,f,"glViewport");
	get_logger()[log::info] << "glViewport (" << x << ", " << y << ", " << width << ", " << height << ")" ;
	get_wrap_builder().set_viewport({static_cast<dimension_t>(width), static_cast<dimension_t>(height), x, y});
	if (f)  f(x, y, width, height );
}

typedef void (*SDL_GL_SwapBuffers_t)(void);

void SDL_GL_SwapBuffers()
{
	GET_FUNC_PTR(SDL_GL_SwapBuffers_t,f,"SDL_GL_SwapBuffers");
	get_logger()[log::verbose_debug] << "SDL_GL_SwapBuffers()";
	get_wrap_builder().pre_swap();
	if (f) f();
	get_wrap_builder().post_swap();
}

}


//extern "C" void *__libc_dlsym  (void *__map, __const char *__name);
//
//void *dlsym(void *handle, const char *symbol)
//{
//
////	get_logger()[log::info] << "Request for " << symbol;
//
//	if (!strcmp(symbol,"dlsym")) return (void*)&dlsym;
//	if (!strcmp(symbol,"glViewport")) return (void*)&glViewport;
//	if (!strcmp(symbol,"glXSwapBuffers")) return (void*)&glXSwapBuffers;
//	if (!strcmp(symbol,"SDL_GL_SwapBuffers")) return (void*)&glXSwapBuffers;
//	if (!strcmp(symbol,"__neko__glViewport")) return __libc_dlsym(handle,"glViewport");
//	if (!strcmp(symbol,"__neko__glXSwapBuffers")) return __libc_dlsym(handle,"glXSwapBuffers");
//	if (!strcmp(symbol,"__neko__SDL_GL_SwapBuffers")) return __libc_dlsym(handle,"SDL_GL_SwapBuffers");
//	return (*__libc_dlsym)(handle,symbol);
//}


void *get_ptr(const std::string& name)
{
	return get_wrap_builder().get_func(name);
}

std::string get_config_name()
{
	if (auto value = getenv("YURI_WRAPPER_CONFIG")) {
		return value;
	}
	return "wrapper.xml";
}
std::string get_grabber_name()
{
	if (auto value = getenv("YURI_WRAPPER_NODE")) {
		return value;
	}
	return "grabber";
}

