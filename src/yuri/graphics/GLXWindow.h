/*!
 * @file 		GLXWindow.h
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef GLXWINDOW_H_
#define GLXWINDOW_H_
//#include "yuri/config/Config.h"
#include "WindowBase.h"

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#ifndef GLX_GLXEXT_PROTOTYPES
#define GLX_GLXEXT_PROTOTYPES

#include <GL/gl.h>
#include <GL/glx.h>
#include <map>
#include <boost/shared_ptr.hpp>

//define GLXWINDOW_USING_GLOBAL_MUTEX

namespace yuri
{
namespace graphics
{

class GLXWindow: public WindowBase
{
protected:
	static std::map<GLXContext,shared_ptr<GLXWindow> > used_contexts;
	static yuri::mutex contexts_map_mutex;
public:
	static void add_used_context(GLXContext ctx,shared_ptr<GLXWindow> win);
	static void remove_used_context(GLXContext ctx);
	static bool is_context_used(GLXContext ctx);


	GLXWindow(log::Log &log_, core::pwThreadBase parent, core::Parameters &p);
	virtual ~GLXWindow();
	static core::pParameters configure();

	virtual bool create();
	virtual bool create_window();
	virtual void setHideDecoration(bool value);
	virtual void addAttributes(int no,GLint *attrs);
	virtual void addAttribute(GLint attr);
	virtual void show(bool /*value*/=true);

	inline Display *getDisplay() { return display.get(); }
	inline Window getWindow() { return win; }
	virtual void swap_buffers();
	virtual bool process_events();
	std::string get_keyname(int key);
	virtual bool check_key(int keysym);
	virtual void exec(core::pCallback c);
	virtual bool have_stereo();
	virtual bool set_vsync(bool state);
protected:
	void initAttr();
	bool load_config();
	void move();
	virtual void do_move();
	virtual void do_show();
	std::string do_get_keyname(int key);
	bool resize(unsigned int w, unsigned int h);

	std::string 			screen;
	shared_ptr<Display>		display;
	Window					root, win;
	std::vector<GLint>		attributes;
	int 					noAttr;
	XVisualInfo*			vi;
	Colormap				cmap;
	XSetWindowAttributes    swa;
	GLXContext				glc;
	XWindowAttributes       gwa;
	XEvent                  xev;
	bool 					override_redirect;
	bool 					hideDecoration;
	bool 					use_stereo;
	bool 					show_cursor;
	int 					screen_number;
	std::string				winname;
#ifdef GLXWINDOW_USING_GLOBAL_MUTEX
	static yuri::mutex	global_mutex;
#else
	yuri::mutex	local_mutex;
#endif
	bool vsync;
};
/*
struct VIDeleter{
	VIDeleter(shared_ptr<Display> d):d(d) {}
	void operator()(XVisualInfo*) {}
	shared_ptr<Display> d;
};
*/
struct DisplayDeleter{
	void operator()(Display*d) { XCloseDisplay(d); }
};
}
}
#endif
#endif /*GLXWINDOW_H_*/
