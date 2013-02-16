/*!
 * @file 		GLXWindow.cpp
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "GLXWindow.h"
#include "yuri/graphics/GL.h"
#include <X11/XKBlib.h>
#include <GL/glxext.h>

using namespace std;
namespace yuri
{
namespace graphics
{
using namespace yuri::log;
using yuri::threads::ThreadBase;
#ifdef GLXWINDOW_USING_GLOBAL_MUTEX
boost::mutex	GLXWindow::global_mutex;
#endif

map<GLXContext,shared_ptr<GLXWindow> > GLXWindow::used_contexts;
boost::mutex GLXWindow::contexts_map_mutex;

GLXWindow::GLXWindow(Log &log_, pThreadBase parent, Parameters &p)
	:WindowBase(log_, parent, p),win(0),noAttr(0),vi(0),
	override_redirect(false),hideDecoration(true),screen_number(0),vsync(false)
{
	shared_ptr<Parameters> def_params = configure();
	params.merge(*def_params);
	params.merge(p);
	log.setLabel("[GLX] ");
}

GLXWindow::~GLXWindow()
{
	if (display) {
		glXDestroyContext(display.get(),glc);
		XDestroyWindow(display.get(),win);
		//XCloseDisplay(display);
		display.reset();
	}
	log[debug] << "GLXWindow::~GLXWindow" << std::endl;
}
shared_ptr<Parameters> GLXWindow::configure()
{
	shared_ptr<Parameters> p(new Parameters());
	(*p)["display"]=":0.0";
	(*p)["stereo"]=false;
	(*p)["width"]=640;
	(*p)["height"]=480;
	(*p)["x"]=0;
	(*p)["y"]=0;
	(*p)["cursor"]=false;
	return p;
}
void GLXWindow::initAttr()
{
	GLint defatr[]= {GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER};
	addAttributes(4, defatr);
	if (use_stereo) addAttribute(GLX_STEREO);
}


void GLXWindow::addAttributes(int no, GLint *attrs)
{
	if (attributes) {
		shared_array<GLint> newAttr(new GLint [noAttr+no]);
		for (int i=0;i<noAttr-1;++i) newAttr[i]=attributes[i];
		for (int i=0;i<no;++i) newAttr[noAttr+i-1]=attrs[i];
		noAttr+=no;
		newAttr[noAttr-1]=None;
		//delete attributes;
		attributes=newAttr;
	} else {
		attributes.reset(new GLint [no+1]);
		for (int i=0;i<no;++i) attributes[i]=attrs[i];
		attributes[no]=None;
		noAttr=no+1;
	}
}

void GLXWindow::addAttribute(GLint attr)
{
	addAttributes(1,&attr);
}
/*
#define check_opt(o) s=prefix+"."+(o); if (!config->exists(s)) \
	{ log[error]<< "Option " << prefix << "." << (o) << " is not present!" \
	<< std::endl; return false;}

#define load_opt(o,v) s=prefix+"."+(o); if (config->get_value(s,v)) \
	log[verbose_debug] << s << " is " << v << std::endl; else { \
	log[error] << "Error while loading " << (o) << std::endl; return false;}
*/
bool GLXWindow::load_config()
{
//	yuri::config::Config *config=yuri::config::Config::get_config();
//	if (!config) {
//		log[error] << "Can't get config!" << std::endl;
//		return false;
//	}
	/*
	// Check for required settings
	std::string s;
	log[debug] << "Checking config for required options" << std::endl;
	check_opt("screen")
	check_opt("x")
	check_opt("y")
	check_opt("width")
	check_opt("height")


	load_opt("x",x)
	load_opt("y",y)
	load_opt("width",width)
	load_opt("height",height)
	load_opt("screen",screen);
	load_opt("stereo",use_stereo);

	config->get_value(prefix+".name",winname,prefix);
*/
	screen = params["display"].get<std::string>();
	use_stereo = params["stereo"].get<bool>();
	show_cursor = params["cursor"].get<bool>();
	x = params["x"].get<yuri::ssize_t>();
	y = params["y"].get<yuri::ssize_t>();
	width = params["width"].get<yuri::size_t>();
	height = params["height"].get<yuri::size_t>();
	//std::string s = params["key_callback"].get<std::string>();
	//keyCallback = config->get_callback(s);
	return true;
}

bool GLXWindow::create_window()
{
	display.reset(XOpenDisplay(screen.c_str()),DisplayDeleter());
	if (!display) return false;
	log[debug] << "Connected to display " << screen << std::endl;
	std::string::size_type ind=screen.find_last_of(':');
	if (ind!=std::string::npos) {
		ind = screen.find_first_of('.',ind);
		if (ind != std::string::npos) {
			screen_number=atoi(screen.substr(ind+1).c_str());
		}
	}
	log[debug] << "Screen number is " << screen_number << std::endl;
	root=RootWindow(display.get(),screen_number);
	if (!root) return false;
	log[debug] << "Found root window" << std::endl;
	vi = glXChooseVisual(display.get(), screen_number, attributes.get());
	if (!vi) return false;
	log[debug] << "Found visual " << vi->visualid << std::endl;
	cmap = XCreateColormap(display.get(), root, vi->visual, AllocNone);
	swa.colormap = cmap;
	swa.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask
					| KeyReleaseMask;//ResizeRedirectMask;
	swa.border_pixel = 0;
	swa.background_pixel = 0;

	win = XCreateWindow(display.get(), root, x, y, width, height, 0, vi->depth,
			InputOutput, vi->visual, CWBackPixel | CWBorderPixel |CWColormap
			| CWEventMask, &swa);
	log[debug] << "X Window Created" << std::endl;
	return true;
}
bool GLXWindow::create()
{
	log[debug] << "creating GLX window" << std::endl;
	if (!load_config()) return false;
	initAttr();
#ifdef GLXWINDOW_USING_GLOBAL_MUTEX
	boost::mutex::scoped_lock l(global_mutex);
#else
	boost::mutex::scoped_lock l(local_mutex);
#endif
	if (!create_window()) return false;
	setHideDecoration(hideDecoration);
	XStoreName(display.get(),win,winname.c_str());
	boost::mutex::scoped_lock bgl(GL::big_gpu_lock);
	glc = glXCreateContext(display.get(), vi, NULL, GL_TRUE);
	glXMakeCurrent(display.get(), win, glc);
	log[debug] << "Created GLX Context" << std::endl;
	bgl.unlock();
	//yuri::config::Config *config=yuri::config::Config::get_config();
	//if (config) config->get_value(prefix+".cursor",show_cursor,false);
	log[debug] << "Cursor " << (show_cursor ? "will" : "won't") << " be shown"
			<< std::endl;
	if (!show_cursor) {
		log[debug] << "Creating cursor" << std::endl;
		Pixmap pixmap;
		Cursor cursor;
		XColor color;
		// Create 1x1 1bpp pixmap
		pixmap = XCreatePixmap(display.get(), win, 1, 1, 1);
		memset((void*) &color, 0, sizeof(XColor));
		cursor = XCreatePixmapCursor(display.get(), pixmap, pixmap, &color, &color,
				0, 0);
		XDefineCursor(display.get(),win,cursor);
	}
	add_used_context(glc,dynamic_pointer_cast<GLXWindow>(get_this_ptr().lock()));
	do_move();
	do_show();

	return true;
}
void GLXWindow::setHideDecoration(bool value)
{
	if (value) {
#ifdef YURI_HAVE_MOTIF
		MotifWmHints hints;
		Atom mh = None;
		mh=XInternAtom(display.get(),"_MOTIF_WM_HINTS",0);
		hints.flags = MWM_HINTS_DECORATIONS;
		hints.decorations = 0;
		hints.functions = 0;
		hints.input_mode = 0;
		hints.status = 0;
		int r = XChangeProperty(display.get(), win,mh, mh, 32,PropModeReplace,
				(unsigned char *) &hints, 4);
		log[info] << "XChangeProperty returned " << r << std::endl;
#else
		log[debug] << "For hiding WM decoration recompile with USE_XM_H" << std::endl;
#endif
	}

}

void GLXWindow::swap_buffers()
{
#ifdef GLXWINDOW_USING_GLOBAL_MUTEX
	boost::mutex::scoped_lock l(global_mutex);
#else
	boost::mutex::scoped_lock l(local_mutex);
#endif
	assert(display);
	assert(win);
	glXSwapBuffers(display.get(), win);
}

void GLXWindow::show(bool /*value*/)
{
#ifdef GLXWINDOW_USING_GLOBAL_MUTEX
	boost::mutex::scoped_lock l(global_mutex);
#else
	boost::mutex::scoped_lock l(local_mutex);
#endif
	do_show();
}

void GLXWindow::move()
{
#ifdef GLXWINDOW_USING_GLOBAL_MUTEX
	boost::mutex::scoped_lock l(global_mutex);
#else
	boost::mutex::scoped_lock l(local_mutex);
#endif
	do_move();
}
void GLXWindow::do_show()
{
	assert(display);
	assert(win);
	XMapWindow(display.get(), win);
	XMoveWindow(display.get(), win,x,y);
}
void GLXWindow::do_move()
{
	assert(display);
	assert(win);
	XMoveWindow(display.get(), win,x,y);
	XRaiseWindow(display.get(),win);
}

bool GLXWindow::process_events()
{
#ifdef GLXWINDOW_USING_GLOBAL_MUTEX
	boost::mutex::scoped_lock l(global_mutex);
#else
	boost::mutex::scoped_lock l(local_mutex);
#endif
	assert(display);
	assert(win);
	if (XCheckWindowEvent(display.get(),win,StructureNotifyMask|KeyPressMask
			|KeyReleaseMask,&xev))
	{
		switch (xev.type) {
		case DestroyNotify:
			log[debug] << "DestroyNotify received" << std::endl;
			request_end();
			//parent->stop();
			break;
		case ConfigureNotify:
			if (resize(xev.xconfigure.width,xev.xconfigure.height))
				glViewport(0,0,xev.xconfigure.width,xev.xconfigure.height);
			break;
		case KeyPress:
			log[debug] << "Key " << do_get_keyname(xev.xkey.keycode) << " (" <<
				xev.xkey.keycode << ") pressed" <<std::endl;
			keys[xev.xkey.keycode]=true;
			// TODO: need to reenable this again!
			//if (keyCallback) keyCallback->run(&xev.xkey.keycode);
			//if (xev.xkey.keycode==9) request_end();
			break;
		case KeyRelease:
			log[debug] << "Key " << do_get_keyname(xev.xkey.keycode) << " (" <<
				xev.xkey.keycode << ") released" <<std::endl;
			keys[xev.xkey.keycode]=false;
			break;
		}
		return true;
	}
	return false;
}

bool GLXWindow::resize(unsigned int w, unsigned int h)
{
	if (w!=(unsigned int)width || h!=(unsigned int)height) {
		log[debug] << "Window size changed! " << width << "x" << height <<
			" -> " << w << "x" << h << std::endl;
		width=w;
		height=h;
		return true;
	}
	return false;

}

std::string GLXWindow::get_keyname(int key)
{
#ifdef GLXWINDOW_USING_GLOBAL_MUTEX
	boost::mutex::scoped_lock l(global_mutex);
#else
	boost::mutex::scoped_lock l(local_mutex);
#endif
	return do_get_keyname(key);
}
std::string GLXWindow::do_get_keyname(int key)
{
	assert(display);
	assert(win);
	std::string keyname = XKeysymToString(XkbKeycodeToKeysym(display.get(), key, 0, 0));
	//std::string keyname = XKeysymToString(XKeycodeToKeysym(display.get(), key, 0));
	return keyname;
}

bool GLXWindow::check_key(int keysym)
{
	boost::mutex::scoped_lock l(keys_lock);
	assert(display);
	assert(win);
	if (keys.find(keysym)==keys.end()) return false;
	return keys[keysym];
}

void GLXWindow::exec(shared_ptr<yuri::config::Callback> c)
{
	boost::mutex::scoped_lock l(keys_lock);
	assert(display);
	assert(win);
	if (c) c->run(get_this_ptr());
}

bool GLXWindow::have_stereo()
{
	return use_stereo;
}

void GLXWindow::add_used_context(GLXContext ctx,shared_ptr<GLXWindow> win)
{
	boost::mutex::scoped_lock l(contexts_map_mutex);
	used_contexts[ctx]=win;
}
void GLXWindow::remove_used_context(GLXContext ctx)
{
	boost::mutex::scoped_lock l(contexts_map_mutex);
	used_contexts.erase(ctx);
}
bool GLXWindow::is_context_used(GLXContext ctx)
{
	boost::mutex::scoped_lock l(contexts_map_mutex);
	return (bool)(used_contexts.find(ctx) != used_contexts.end());
}

bool GLXWindow::set_vsync(bool state)
{
#ifdef YURI_HAVE_vsync
#ifdef __linux__
#ifdef GLX_SGI_swap_control
	glXSwapIntervalSGI(state?1:0);
	return true;
#else
	return false;
#endif
#else
	return false;
#endif
#else
	(void)state;
	return false;
#endif
}
} // End of graphics
} // End of yuri

//End of File
