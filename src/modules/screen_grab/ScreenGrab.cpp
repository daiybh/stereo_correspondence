/*
 * ScreenGrab.cpp
 *
 *  Created on: 7.3.2013
 *      Author: neneko
 */
#include "ScreenGrab.h"
#include "yuri/core/Module.h"
#include "X11/Xutil.h"
namespace yuri {
namespace screen {

REGISTER("screen",ScreenGrab)
IO_THREAD_GENERATOR(ScreenGrab)

core::pParameters ScreenGrab::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("ScreenGrab module.");
	(*p)["display"]["X display"]=std::string(":0");
	(*p)["fps"]["Frames per second"]=10.0;
	(*p)["x"]["X offset of the grabbed image"]=0;
	(*p)["y"]["Y offset of the grabbed image"]=0;
	(*p)["width"]["Width of the grabbed image (set to -1 to grab full image)"]=-1;
	(*p)["height"]["Height of the grabbed image (set to -1 to grab full image)"]=-1;
	p->set_max_pipes(1,1);
	return p;
}
namespace {
struct DisplayDeleter{
	void operator()(Display*d) { XCloseDisplay(d); }
};
struct ImageDeleter{
	void operator()(XImage*i) { XDestroyImage(i); }
};
}

ScreenGrab::ScreenGrab(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,1,1,std::string("screen_grab")),win(0),x(0),y(0),
width(-1),height(-1)
{
	IO_THREAD_INIT("ScreenGrab")
	dpy.reset(XOpenDisplay(display.c_str()),DisplayDeleter());
	if (!dpy) {
		throw exception::InitializationFailed("Failed to open connection to X display at '"+display+"'");
	}
	log[log::info] << "Connected to display " << display;
	win = DefaultRootWindow(dpy.get());

}

ScreenGrab::~ScreenGrab()
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
	IO_THREAD_PRE_RUN
	while(still_running()) {
		if (!grab()) break;
	}

	IO_THREAD_POST_RUN
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
	format_t fmt = YURI_FMT_NONE;
	switch (img->bits_per_pixel) {
		case 32: fmt = YURI_FMT_BGRA; break;
		case 24: fmt = YURI_FMT_BGR; break;
	}
	if (fmt!=YURI_FMT_NONE) {
		core::pBasicFrame frame = allocate_empty_frame(fmt, w,h, true);
		const ubyte_t* data = reinterpret_cast<ubyte_t*>(img->data);
		ubyte_t *out = PLANE_RAW_DATA(frame,0);
		const size_t copy_bytes = w*img->bits_per_pixel/8;
		for (int line=0;line<h;++line) {
			std::copy(data+img->bytes_per_line*line,data+img->bytes_per_line*line+copy_bytes,out);
			out+=copy_bytes;
		}
		push_raw_video_frame(0,frame);
	}
	return true;
}

bool ScreenGrab::set_param(const core::Parameter &param)
{
	if (param.name == "display") {
		display = param.get<std::string>();
	} else if (param.name == "fps") {
		fps = param.get<double>();
	} else if (param.name == "x") {
		x = param.get<ssize_t>();
	} else if (param.name == "y") {
		y = param.get<ssize_t>();
	} else if (param.name == "width") {
		width = param.get<ssize_t>();
	} else if (param.name == "height") {
		height = param.get<ssize_t>();
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace screen */
} /* namespace yuri */


