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
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include <unordered_map>
namespace yuri {
namespace sdl_window {

namespace {
std::unordered_map<format_t, Uint32> yuri_to_sdl_yuv =
	{{core::raw_format::yuyv422, SDL_YUY2_OVERLAY},
	 {core::raw_format::yvyu422, SDL_YVYU_OVERLAY},
	 {core::raw_format::uyvy422, SDL_UYVY_OVERLAY}};
// TODO: SUpport for planar SDL_YV12_OVERLAY and SDL_IYUV_OVERLAY

Uint32 map_yuv_yuri_to_sdl(format_t fmt) {
	auto it = yuri_to_sdl_yuv.find(fmt);
	if (it == yuri_to_sdl_yuv.end()) return 0;
	return it->second;
}
}


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
	p["opengl"]["Use OpenGL for rendering"]=false;
	p["default_keys"]["Enable default key events. This includes ESC for quit and f for fullscreen toggle."]=true;
	return p;
}


SDLWindow::SDLWindow(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,0,std::string("sdl_window")),
resolution_({800,600}),fullscreen_(false),default_keys_(true),use_gl_(false)
{
	IOTHREAD_INIT(parameters)
	set_latency(10_ms);

	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE) < 0) {
		throw exception::InitializationFailed("Failed to initialize SDL");
	}
	if (!(surface_ = SDL_SetVideoMode(resolution_.width, resolution_.height, 24,
		      SDL_HWSURFACE |  SDL_DOUBLEBUF | SDL_RESIZABLE |
		      (fullscreen_?SDL_FULLSCREEN:0) |
		      (use_gl_?SDL_OPENGL:0)))) {
		throw exception::InitializationFailed("Failed to set video mode");
	}

}

SDLWindow::~SDLWindow()
{
	SDL_Quit();
}

bool SDLWindow::step()
{
	process_sdl_events();
	core::pFrame gframe = pop_frame(0);
	core::pRawVideoFrame frame = dynamic_pointer_cast<core::RawVideoFrame>(gframe);
	if (!frame) return true;
	const resolution_t res = frame->get_resolution();
	format_t format = frame->get_format();
	Uint32 sdl_fmt = map_yuv_yuri_to_sdl(format);
	if (sdl_fmt) {
		if (!overlay_ ||
				overlay_->w != static_cast<int>(res.width) ||
				overlay_->h != static_cast<int>(res.height) ||
				overlay_->format != sdl_fmt) {
			overlay_.reset(SDL_CreateYUVOverlay(res.width, res.height, sdl_fmt, surface_),[](SDL_Overlay*o){SDL_FreeYUVOverlay(o);});
		}
		if (!overlay_) {
			log[log::error] << "Failed to allocate overlay";
			return false;
		}
		dimension_t src_linesize  = PLANE_DATA(frame,0).get_line_size();
		dimension_t target_linesize  = static_cast<dimension_t>(overlay_->pitches[0]);
		dimension_t copy_linesize = std::min(src_linesize, target_linesize);
		auto it = PLANE_DATA(frame,0).begin();
		SDL_LockYUVOverlay(overlay_.get());
		for (dimension_t line = 0; line < res.height; ++line) {
			std::copy(it, it + copy_linesize, overlay_->pixels[0] + target_linesize * line);
			it += src_linesize;
		}
		SDL_UnlockYUVOverlay(overlay_.get());
		SDL_Rect rec={0,0,static_cast<Uint16>(resolution_.width), static_cast<Uint16>(resolution_.height)};
		SDL_DisplayYUVOverlay(overlay_.get(), &rec);
	} else {
		const auto& fi = core::raw_format::get_format_info(format);
		log[log::warning] << "Unsupported format '" << fi.name << "'";
	}

	return true;
}
bool SDLWindow::set_param(const core::Parameter& param)
{
	if (iequals(param.get_name(), "resolution")) {
		resolution_ = param.get<resolution_t>();
	} else if (iequals(param.get_name(), "fullscreen")) {
		fullscreen_ = param.get<bool>();
	} else if (iequals(param.get_name(), "default_keys")) {
		default_keys_ = param.get<bool>();
	} else if (iequals(param.get_name(), "opengl")) {
		use_gl_ = param.get<bool>();
	} else return core::IOThread::set_param(param);
	return true;
}
void SDLWindow::process_sdl_events()
{
	SDL_Event event;
	while(SDL_PollEvent(&event)) {
		switch (event.type) {
			case SDL_QUIT: request_end(core::yuri_exit_interrupted);
				break;
			case SDL_VIDEORESIZE:
				sdl_resize({static_cast<dimension_t>(event.resize.w), static_cast<dimension_t>(event.resize.h)});
				break;
			case SDL_KEYDOWN:
				if (default_keys_) {
					if (event.key.keysym.sym == SDLK_ESCAPE) request_end(core::yuri_exit_interrupted);
					if (event.key.keysym.sym == SDLK_f) { fullscreen_=!fullscreen_; sdl_resize(resolution_);}
				}
				break;

			default:break;
		}
	}
}
void SDLWindow::sdl_resize(resolution_t new_res)
{
	resolution_ = new_res;
	if (!use_gl_) {
		overlay_.reset();
		Uint32 flags = (surface_->flags & ~SDL_FULLSCREEN) | (fullscreen_?SDL_FULLSCREEN:0);
		if (surface_) surface_ = SDL_SetVideoMode(resolution_.width, resolution_.height, 24, flags);
	}
}
} /* namespace sdl_window */
} /* namespace yuri */
