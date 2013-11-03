/*
 * ScreenGrab.cpp
 *
 *  Created on: 7.3.2013
 *      Author: neneko
 */
#include "ScreenGrab.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "X11/Xutil.h"
#include "X11/Xatom.h"
#include <string>
namespace yuri {
namespace screen {


IOTHREAD_GENERATOR(ScreenGrab)

MODULE_REGISTRATION_BEGIN("screen")
		REGISTER_IOTHREAD("screen",ScreenGrab)
MODULE_REGISTRATION_END()

core::Parameters ScreenGrab::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("ScreenGrab module.");
	p["display"]["X display"]=std::string();
	p["fps"]["Frames per second"]=10.0;
	p["x"]["X offset of the grabbed image"]=0;
	p["y"]["Y offset of the grabbed image"]=0;
	p["width"]["Width of the grabbed image (set to -1 to grab full image)"]=-1;
	p["height"]["Height of the grabbed image (set to -1 to grab full image)"]=-1;
	p["win_name"]["Window name (set to empty string to grab whole screen)"]=std::string();
	p["pid"]["PID of application that created the window (set to 0 to grab whole screen)"]=0;
	p["win_id"]["Window ID (set to 0 to grab whole screen)"]=0;
//	p->set_max_pipes(1,1);
	return p;
}
namespace {
struct DisplayDeleter{
	void operator()(Display*d) { XCloseDisplay(d); }
};
struct ImageDeleter{
	void operator()(XImage*i) { XDestroyImage(i); }
};
std::string get_win_name(Display* dpy, Window win)
{
	std::string str;
	char *win_name;
	XFetchName(dpy, win, &win_name);
	if (win_name) {
		str = win_name;
		XFree(win_name);
	}
	return str;
}
size_t get_win_pid(Display* dpy, Window win)
{
	Atom atom = XInternAtom(dpy, "_NET_WM_PID", True);
	if (atom==None) return 0;
	Atom           type;
	int            format;
	unsigned long  nItems;
	unsigned long  bytesAfter;
	unsigned char *pid_ = 0;
	size_t 	pid = 0;
	if(XGetWindowProperty(dpy, win, atom, 0, 1, False, XA_CARDINAL, &type, &format, &nItems, &bytesAfter, &pid_)==Success)
	{
		if(pid_ != 0)
		{
			pid = *reinterpret_cast<unsigned long *>(pid_);
			XFree(pid_);
		}
	}
	return pid;
}
// Dummy method just to be able to use find)child bellow
Window get_win_id(Display* /*dpy*/, Window win) {
	return win;
}
template<typename T, typename F>
Window find_child(Display* dpy, Window top, T val, F func)
{
	if (func(dpy, top)==val) return top;
	Window dummy_win;
	Window *childs;
	Window found_win = 0;
	unsigned int child_count;
	XQueryTree(dpy,top,&dummy_win,&dummy_win,&childs,&child_count);
	if (childs && child_count) {
		for (unsigned int i=0; i<child_count;++i) {
			if((found_win = find_child(dpy,childs[i],val,func))!=0) break;
		}
	}
	XFree(childs);
	return found_win;
}
}

ScreenGrab::ScreenGrab(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("screen_grab")),win(0),x(0),y(0),
width(-1),height(-1),pid(0),win_id_(0)
{
	IOTHREAD_INIT(parameters)
	XInitThreads();
	dpy.reset(XOpenDisplay(display.c_str()),DisplayDeleter());
	if (!dpy) {
		throw exception::InitializationFailed("Failed to open connection to X display at '"+display+"'");
	}
	log[log::info] << "Connected to display " << display;
	win = DefaultRootWindow(dpy.get());
	if (win_id_ != 0) {
		win = find_child(dpy.get(),win,win_id_,get_win_id);
		if (!win) {
			throw exception::InitializationFailed("Failed to find window with specified ID");
		}
	} else if (!win_name.empty()) {
		win = find_child(dpy.get(),win,win_name,get_win_name);
		if (!win) {
			throw exception::InitializationFailed("Failed to find window "+win_name);
		}
	} else if (pid) {
		win = find_child(dpy.get(),win,pid,get_win_pid);
		if (!win) {
			throw exception::InitializationFailed("Failed to find window for specified PID");
		}
	}
	log[log::info] << "Grabbing window " << get_win_name(dpy.get(), win);

}

ScreenGrab::~ScreenGrab() noexcept
{
}

/*bool ScreenGrab::step()
{	if (!in[0]) return true;
	core::pBasicFrame frame = in[0]->pop_frame();
	if (!frame) return true;

	//push_raw_frame(0,frame);
	return true;
}*/

void ScreenGrab::run()
{
//	IO_THREAD_PRE_RUN
	while(still_running()) {
		if (!grab()) break;
	}
//	IO_THREAD_POST_RUN
}
namespace {

}
bool ScreenGrab::grab()
{
	XWindowAttributes attr;
	XGetWindowAttributes(dpy.get(),win,&attr);
	log[log::debug] << "Found window " << attr.width << "x" << attr.height;
	if (x > attr.width && y > attr.height) {
		log[log::warning] << "Offset out of range of the image";
		return true;
	}
	const int w = (width>0)?std::min(attr.width-x,width):attr.width-x;
	const int h = (height>0)?std::min(attr.height-y,height):attr.height-y;

	shared_ptr<XImage> img (XGetImage(dpy.get(),win,x,y,w,h,AllPlanes,ZPixmap),ImageDeleter());
	if (!img) {
		log[log::warning] << "Failed to get image from the window";
		return true;
	}
	log[log::debug] << "Image has depth " << img->depth << ", bpl: " << img->bytes_per_line << ", bpp: " << img->bits_per_pixel;
	format_t fmt = 0;
	switch (img->bits_per_pixel) {
		case 32: fmt = core::raw_format::bgra32; break;
		case 24: fmt = core::raw_format::bgr24; break;
	}
	if (fmt!=0) {
		resolution_t res = {static_cast<dimension_t>(w), static_cast<dimension_t>(h)};
		core::pRawVideoFrame frame = core::RawVideoFrame::create_empty(fmt, res, true);
		const uint8_t* data = reinterpret_cast<uint8_t*>(img->data);
		uint8_t *out = PLANE_RAW_DATA(frame,0);
		const size_t copy_bytes = w*img->bits_per_pixel/8;
		for (int line=0;line<h;++line) {
			std::copy(data+img->bytes_per_line*line,data+img->bytes_per_line*line+copy_bytes,out);
			out+=copy_bytes;
		}
		push_frame(0,frame);
	}
	return true;
}

bool ScreenGrab::set_param(const core::Parameter &param)
{
	if (param.get_name() == "display") {
		display = param.get<std::string>();
	} else if (param.get_name() == "fps") {
		fps = param.get<double>();
	} else if (param.get_name() == "x") {
		x = param.get<ssize_t>();
	} else if (param.get_name() == "y") {
		y = param.get<ssize_t>();
	} else if (param.get_name() == "width") {
		width = param.get<ssize_t>();
	} else if (param.get_name() == "height") {
		height = param.get<ssize_t>();
	} else if (param.get_name() == "win_name") {
		win_name = param.get<std::string>();
	} else if (param.get_name() == "pid") {
		pid = param.get<size_t>();
	} else if (param.get_name() == "win_id") {
		win_id_ = param.get<Window>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace screen */
} /* namespace yuri */


