/*
 * wrapper.cpp
 *
 *  Created on: Oct 5, 2010
 *      Author: worker
 */


#include "yuri/core/ApplicationBuilder.h"
#include "../modules/fb_grab/FrameBufferGrabber.h"
#include "yuri/core/Module.h"
extern "C" {
	#include "GL/glx.h"
#ifdef YURI_HAVE_SDL
	#include <SDL_video.h>
#endif
}
void __attribute__ ((constructor)) __yuri_wrapper_loader(void);
void *get_ptr(std::string name);

#define GET_FUNC_PTR(type,ptr,name) type ptr=reinterpret_cast<type>(reinterpret_cast<uintptr_t>(get_ptr(name)));

using namespace yuri;


class __yuri_starter:public yuri::core::IOThread {
public:
	__yuri_starter(log::Log &l);
	shared_ptr<core::ApplicationBuilder> builder;
	shared_ptr<fbgrab::FrameBufferGrabber> fbgrab;
	shared_ptr<fbgrab::FrameBufferGrabber> get_grabber() {
		//return dynamic_pointer_cast<fbgrab::FrameBufferGrabber>(builder->get_node(grabber_name));
		return yuri::static_pointer_cast<fbgrab::FrameBufferGrabber>(builder->get_node(grabber_name));
		//return builder->get_node(grabber_name);
		//return shared_ptr<fbgrab::FrameBufferGrabber>();
	}
	std::string config_file;
	std::string grabber_name;
};


static shared_ptr<log::Log> l;

static shared_ptr<__yuri_starter> __yuri_start;
//static shared_ptr<FrameBufferGrabber> fbgrab;
// We need plain pointer here, otherwise the value might get overwritten
static fbgrab::FrameBufferGrabber* fbgrab_instance=0;


typedef void (*glXSwapBuffers_t)(Display * dpy,	GLXDrawable drawable);

void glXSwapBuffers(Display * dpy,	GLXDrawable drawable)
{
	GET_FUNC_PTR(glXSwapBuffers_t, f, "glXSwapBuffers");
	if (fbgrab_instance) fbgrab_instance->pre_swap();
	if (f) f(dpy,drawable);
	if (fbgrab_instance) fbgrab_instance->post_swap();
}


typedef void (*glViewport_t)(GLint x, GLint y, GLsizei width, GLsizei height);

void glViewport(GLint x, GLint y, GLsizei width, GLsizei height)
{
//	if (!fbgrab_instance) {
//		if (!__yuri_start) __yuri_start.reset(new __yuri_starter(*l));
//		fbgrab_instance = __yuri_start->get_grabber().get();
//		if (!fbgrab_instance) (*l)[log::error] <<"no grabber";
//		else (*l)[log::info] << "Got grabber";
//	}
	GET_FUNC_PTR(glViewport_t,f,"glViewport");
	if (fbgrab_instance) fbgrab_instance->set_viewport(x,y,width,height);
	if (f)  f(x, y, width, height );
}

#ifdef YURI_HAVE_SDL

typedef void (*SDL_GL_SwapBuffers_t)(void);

void SDL_GL_SwapBuffers()
{
	GET_FUNC_PTR(SDL_GL_SwapBuffers_t,f,"SDL_GL_SwapBuffers");
	if (fbgrab_instance) fbgrab_instance->pre_swap();
	if (f) f();
	if (fbgrab_instance) fbgrab_instance->post_swap();
}
#endif

/*
extern "C" void *__libc_dlsym  (void *__map, __const char *__name);

void *dlsym(void *handle, const char *symbol)
{

	std::cerr << "Request for " << symbol << std::endl;

	if (!strcmp(symbol,"dlsym")) return (void*)&dlsym;
	if (!strcmp(symbol,"glViewport")) return (void*)&glViewport;
	if (!strcmp(symbol,"glXSwapBuffers")) return (void*)&glXSwapBuffers;
	if (!strcmp(symbol,"SDL_GL_SwapBuffers")) return (void*)&glXSwapBuffers;
	if (!strcmp(symbol,"__neko__glViewport")) return __libc_dlsym(handle,"glViewport");
	if (!strcmp(symbol,"__neko__glXSwapBuffers")) return __libc_dlsym(handle,"glXSwapBuffers");
	if (!strcmp(symbol,"__neko__SDL_GL_SwapBuffers")) return __libc_dlsym(handle,"SDL_GL_SwapBuffers");
	return (*__libc_dlsym)(handle,symbol);
}
*/

void __yuri_wrapper_loader(void)
{
	l.reset(new log::Log(std::cerr));
	l->set_label("[YURI Wrapper] ");
	l->set_flags(log::normal|log::show_time);
	(*l)[log::info] << "Wrapper starting";
	if (!__yuri_start) __yuri_start.reset(new __yuri_starter(*l));
			fbgrab_instance = __yuri_start->get_grabber().get();
			if (!fbgrab_instance) (*l)[log::error] <<"no grabber";
			else (*l)[log::info] << "Got grabber";
}

__yuri_starter::__yuri_starter(log::Log &l):core::IOThread(l,core::pwThreadBase(),0,0,"YuriStarter")
{
	if (!core::RegisteredClass::is_registered("appbuilder")) {
		log[log::error] << "No app builder found! This won't work";
		throw exception::Exception("bajbaj");
	}
	log[log::debug] << "Starting __yuri_starter";
	config_file = "wrapper.xml";
	grabber_name = "grabber";
	const char *str = 0;
	if ((str = getenv("YURI_GRAB_CONFIG"))) config_file = str;
	if ((str = getenv("YURI_GRAB_OBJECT"))) grabber_name = str;
	core::pParameters params;
	core::pInstance builder_instance;
	try {
		builder_instance = core::RegisteredClass::prepare_instance("appbuilder");
	}
	catch (exception::Exception &e) {
		log[log::error] << "Exception during prepare_instace";
		builder_instance.reset();
	}
	if (!builder_instance) {
		log[log::error] << "Failed to prepare instance";
		throw exception::Exception("bajbaj");
	}
	params = builder_instance->params;
	(*params)["config"] = config_file;
	try {
		builder = dynamic_pointer_cast<core::ApplicationBuilder>(builder_instance->create_class(l,core::pwThreadBase()));
	}
	catch (std::exception &e) {
		log[log::error] << "Instantiation of app builder threw exception! (" << e.what()<< ")";
		throw exception::Exception("bajbaj");
	}
	builder->prepare_threads();
	spawn_thread(builder);
}

void *get_ptr(std::string name)
{
	return fbgrab_instance?fbgrab_instance->get_function(name):0;
}

