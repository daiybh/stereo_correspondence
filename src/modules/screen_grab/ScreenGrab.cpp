/*
 * ScreenGrab.cpp
 *
 *  Created on: 7.3.2013
 *      Author: neneko
 */
#include "ScreenGrab.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace screen {

REGISTER("pass",ScreenGrab)
IO_THREAD_GENERATOR(ScreenGrab)

core::pParameters ScreenGrab::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("ScreenGrab module.");
	(*p)["display"]["X display"]=std::string(":0");
	(*p)["fps"]["Frames per second"]=10.0;
	p->set_max_pipes(1,1);
	return p;
}
namespace {
struct DisplayDeleter{
	void operator()(Display*d) { XCloseDisplay(d); }
};

}

ScreenGrab::ScreenGrab(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,1,1,std::string("screen_grab")),win(0)
{
	IO_THREAD_INIT("ScreenGrab")
	dpy.reset(XOpenDisplay(display.c_str()),DisplayDeleter());
	if (!dpy) {
		throw exception::InitializationFailed("Failed to open connection to X display at '"+display+"'");
	}
}

ScreenGrab::~ScreenGrab()
{
}

/*bool ScreenGrab::step()
{
	if (!in[0]) return true;
	core::pBasicFrame frame = in[0]->pop_frame();
	if (!frame) return true;

	//push_raw_frame(0,frame);
	return true;
}*/

void ScreenGrab::run()
{
	IO_THREAD_PRE_RUN


	IO_THREAD_POST_RUN
}
bool ScreenGrab::set_param(const core::Parameter &param)
{
	if (param.name == "display") {
		display = param.get<std::string>();
	} else if (param.name == "fps") {
		fps = param.get<double>();
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace screen */
} /* namespace yuri */


