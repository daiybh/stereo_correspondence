/*!
 * @file 		SDLWindow.cpp
 * @author 		<Your name>
 * @date		24.09.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "SDLWindow.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/RawVideoFrame.h"
namespace yuri {
namespace sdl_window {


IOTHREAD_GENERATOR(SDLWindow)

MODULE_REGISTRATION_BEGIN("sdl_window")
		REGISTER_IOTHREAD("sdl_window",SDLWindow)
MODULE_REGISTRATION_END()

core::Parameters SDLWindow::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("SDLWindow");
	p["resolution"]["Resolution of output window"]=resolution_t{800,600};
	p["fullscreen"]["Start in fullscreen"]=false;
	return p;
}


SDLWindow::SDLWindow(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,0,std::string("sdl_window")),
resolution_({800,600}),fullscreen_(false)
{
	IOTHREAD_INIT(parameters)
	set_latency(10_ms);

	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE) < 0) {
		throw exception::InitializationFailed("Failed to initialize SDL");
	}
	if (!(surface_ = SDL_SetVideoMode(resolution_.width, resolution_.height, 24,
		      SDL_HWSURFACE |  SDL_DOUBLEBUF |
		      (fullscreen_?SDL_FULLSCREEN:0) ))) {
		throw exception::InitializationFailed("Failed to set video mode");
	}

}

SDLWindow::~SDLWindow()
{
}

bool SDLWindow::step()
{
	core::pFrame gframe = pop_frame(0);
	core::pRawVideoFrame frame = dynamic_pointer_cast<core::RawVideoFrame>(gframe);
	if (!frame) return true;
	const resolution_t res = frame->get_resolution();
	if (frame->get_format() == core::raw_format::yuyv422) {
		shared_ptr<SDL_Overlay> overlay(SDL_CreateYUVOverlay(res.width, res.height, 0x59565955, surface_),[](SDL_Overlay*o){SDL_FreeYUVOverlay(o);});
//		if (!overlay_) {
//			overlay_ = unique_ptr<SDL_Overlay>(SDL_CreateYUVOverlay(res.width, res.height, 0x59565955, surface_),[](SDL_Overlay*o){SDL_FreeYUVOverlay(o);});
//		}
		if (!overlay) {
			log[log::error] << "Failed to allocate overlay";
			return false;
		}



	}

	return true;
}
bool SDLWindow::set_param(const core::Parameter& param)
{
	if (iequals(param.get_name(), "resolution")) {
		resolution_ = param.get<resolution_t>();
	} else if (iequals(param.get_name(), "fullscreen")) {
		fullscreen_ = param.get<bool>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace sdl_window */
} /* namespace yuri */
