/*!
 * @file 		SDLWindow.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		24.09.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "SDLWindow.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/version.h"
#include "yuri/core/utils/Timer.h"
#include <unordered_map>
namespace yuri {
namespace sdl_window {

namespace {
const std::unordered_map<format_t, Uint32> yuri_to_sdl_yuv =
	{{core::raw_format::yuyv422, SDL_YUY2_OVERLAY},
	 {core::raw_format::yvyu422, SDL_YVYU_OVERLAY},
	 {core::raw_format::uyvy422, SDL_UYVY_OVERLAY}};
// TODO: Support for planar SDL_YV12_OVERLAY and SDL_IYUV_OVERLAY


Uint32 map_yuv_yuri_to_sdl(format_t fmt) {
	auto it = yuri_to_sdl_yuv.find(fmt);
	if (it == yuri_to_sdl_yuv.end()) return 0;
	return it->second;
}

//! Supported formats for HW overlay
const std::vector<format_t> supported_formats = {
	core::raw_format::yuyv422,
	core::raw_format::yvyu422,
	core::raw_format::uyvy422,
	core::raw_format::rgb24,
	core::raw_format::bgr24,
	core::raw_format::rgb15,
	core::raw_format::bgr15,
	core::raw_format::rgb16,
	core::raw_format::bgr16,
	core::raw_format::argb32,
	core::raw_format::rgba32,
	core::raw_format::abgr32,
	core::raw_format::bgra32
};
}


IOTHREAD_GENERATOR(SDLWindow)

MODULE_REGISTRATION_BEGIN("sdl_window")
		REGISTER_IOTHREAD("sdl_window",SDLWindow)
MODULE_REGISTRATION_END()

core::Parameters SDLWindow::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("SDLWindow");
	p["resolution"]["Resolution of output window"]=resolution_t{800,600};
	p["fullscreen"]["Start in fullscreen"]=false;
	p["opengl"]["Use OpenGL for rendering"]=false;
	p["default_keys"]["Enable default key events. This includes ESC for quit and f for fullscreen toggle."]=true;
	p["window_title"]["Window title"]=std::string();
	return p;
}


SDLWindow::SDLWindow(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawVideoFrame>(log_,parent,std::string("sdl_window")),
resolution_({800,600}),fullscreen_(false),default_keys_(true),use_gl_(false),
overlay_{nullptr,[](SDL_Overlay*o){if(o) SDL_FreeYUVOverlay(o);}},
rgb_surface_{nullptr,[](SDL_Surface*s){ if(s) SDL_FreeSurface(s);}},
sdl_bpp_(32),title_(std::string("Yuri2 (")+yuri_version+")"),
gl_(log)
{
	IOTHREAD_INIT(parameters)
	set_latency(1_ms);

	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_NOPARACHUTE) < 0) {
		throw exception::InitializationFailed("Failed to initialize SDL");
	}
	if (!use_gl_) {
		set_supported_formats(supported_formats);
	} else {
		set_supported_formats(gl_.get_supported_formats());
	}
}

SDLWindow::~SDLWindow() noexcept
{
	SDL_Quit();
}

void SDLWindow::run()
{
	print_id();
	if (!(surface_ = SDL_SetVideoMode(resolution_.width, resolution_.height, sdl_bpp_,
			  SDL_HWSURFACE |  SDL_DOUBLEBUF | SDL_RESIZABLE |
			  (fullscreen_?SDL_FULLSCREEN:0) |
			  (use_gl_?SDL_OPENGL:0)))) {
		throw exception::InitializationFailed("Failed to set video mode");
	}
	SDL_WM_SetCaption(title_.c_str(), "yuri2");
	if (use_gl_) {
		gl_.enable_smoothing();
		gl_.setup_ortho();
	}
	IOThread::run();
	overlay_.reset();
	rgb_surface_.reset();
}

bool SDLWindow::step()
{
	process_sdl_events();
	return base_type::step();
}

core::pFrame SDLWindow::do_special_single_step(const core::pRawVideoFrame& frame)
{
	Timer timer;
	const resolution_t res = frame->get_resolution();
	const dimension_t src_linesize  = PLANE_DATA(frame,0).get_line_size();
	auto it = PLANE_DATA(frame,0).begin();

	format_t format = frame->get_format();
	Uint32 sdl_fmt = map_yuv_yuri_to_sdl(format);
	if (use_gl_) {
		glDrawBuffer(GL_BACK_LEFT);

		gl_.generate_texture(0, frame);
		gl_.draw_texture(0);
		gl_.finish_frame();
		SDL_GL_SwapBuffers();
	} else if (sdl_fmt) {
		if (!overlay_ ||
				overlay_->w != static_cast<int>(res.width) ||
				overlay_->h != static_cast<int>(res.height) ||
				overlay_->format != sdl_fmt) {
			overlay_.reset(SDL_CreateYUVOverlay(res.width, res.height, sdl_fmt, surface_));
		}
		if (!overlay_) {
			log[log::error] << "Failed to allocate overlay";
			return {};
		}
		const dimension_t target_linesize  = static_cast<dimension_t>(overlay_->pitches[0]);
		const dimension_t copy_linesize = std::min(src_linesize, target_linesize);
//		SDL_LockYUVOverlay(overlay_.get());
		for (dimension_t line = 0; line < res.height; ++line) {
			std::copy(it, it + copy_linesize, overlay_->pixels[0] + target_linesize * line);
			it += src_linesize;
		}
//		SDL_UnlockYUVOverlay(overlay_.get());
		SDL_Rect rec={0,0,static_cast<Uint16>(resolution_.width), static_cast<Uint16>(resolution_.height)};
		SDL_DisplayYUVOverlay(overlay_.get(), &rec);
	} else if (prepare_rgb_overlay(frame)) {
		const dimension_t target_linesize  = static_cast<dimension_t>(rgb_surface_->pitch);
		const dimension_t copy_linesize = std::min(src_linesize, target_linesize);
		const dimension_t copy_lines = std::min(res.height, resolution_.height);
		for (dimension_t line = 0; line < copy_lines; ++line) {
			std::copy(it, it + copy_linesize, reinterpret_cast<uint8_t*>(rgb_surface_->pixels) + target_linesize * line);
			it += src_linesize;
		}
		SDL_Rect dest_rec={0,0,static_cast<Uint16>(resolution_.width), static_cast<Uint16>(resolution_.height)};
		SDL_Rect src_rec={0,0,static_cast<Uint16>(res.width), static_cast<Uint16>(res.height)};
//		SDL_SoftStretch(rgb_surface_.get(), &src_rec, rgb_surface2_.get(), &dest_rec);
//		SDL_BlitSurface(rgb_surface2_.get(), &dest_rec, surface_, &dest_rec);
//		SDL_SoftStretch(rgb_surface2_.get(), &dest_rec, surface_, &dest_rec);
//		SDL_SoftStretch(rgb_surface_.get(), &src_rec, surface_, &dest_rec);

		SDL_BlitSurface(rgb_surface_.get(), &src_rec, surface_, &dest_rec);
		SDL_Flip(surface_);
	} else {

		const auto& fi = core::raw_format::get_format_info(format);
		log[log::warning] << "Unsupported format '" << fi.name << "'";
	}
	log[log::debug] << "Processing took " << timer.get_duration();
	return {};
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
	} else if (iequals(param.get_name(), "window_title")) {
		std::string new_title = param.get<std::string>();
		if (!new_title.empty()) title_=std::move(new_title);
	} else return base_type::set_param(param);
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
	Uint32 flags = (surface_->flags & ~SDL_FULLSCREEN) | (fullscreen_?SDL_FULLSCREEN:0);
	if (surface_) surface_ = SDL_SetVideoMode(resolution_.width, resolution_.height, sdl_bpp_, flags);
	if (!use_gl_) {
		overlay_.reset();
	} else {
		glViewport( 0, 0, new_res.width, new_res.height );
		gl_.setup_ortho();
	}
}

bool SDLWindow::prepare_rgb_overlay(const core::pRawVideoFrame& frame)
{
	std::tuple<Uint32, Uint32,Uint32, Uint32> masks;
	const resolution_t res = frame->get_resolution();
	format_t format = frame->get_format();
	switch (format) {
		case core::raw_format::rgb24: masks = std::make_tuple(0x0000FF,0x00FF00,0xFF0000,0);
				break;
		case core::raw_format::bgr24: masks = std::make_tuple(0xFF0000,0x00FF00,0x0000FF,0);
				break;
		case core::raw_format::rgb15: masks = std::make_tuple(0xF800,0x07C0,0x001E,0);
				break;
		case core::raw_format::bgr15: masks = std::make_tuple(0x3E00,0x07C0,0x00F8,0);
				break;
		case core::raw_format::rgb16: masks = std::make_tuple(0xF800,0x07E0,0x001F,0);
				break;
		case core::raw_format::bgr16: masks = std::make_tuple(0x1F00,0x07E0,0x00F8,0);
				break;
		case core::raw_format::argb32: masks = std::make_tuple(0x0000FF00,0x00FF0000,0xFF000000,0x00000000);
				break;
		case core::raw_format::rgba32: masks = std::make_tuple(0x000000FF,0x0000FF00,0x00FF0000,0x00000000);
				break;
		case core::raw_format::abgr32: masks = std::make_tuple(0xFF000000,0x00FF0000,0x0000FF00,0x00000000);
				break;
		case core::raw_format::bgra32: masks = std::make_tuple(0x00FF0000,0x0000FF00,0x000000FF,0x00000000);
				break;
		default: return false;
	}
	const auto& fi = core::raw_format::get_format_info(format);
	int bpp = fi.planes[0].bit_depth.first/fi.planes[0].bit_depth.second;
	if (!rgb_surface_ ||
			static_cast<dimension_t>(rgb_surface_->w) != res.width ||
			static_cast<dimension_t>(rgb_surface_->h) != res.height ||
			rgb_surface_->format->BitsPerPixel != bpp) {
		log[log::info] << "(Re)creating RGB surface with " << bpp << " bpp.";
		rgb_surface_.reset(SDL_CreateRGBSurface(SDL_SWSURFACE, res.width, res.height, bpp,
				std::get<0>(masks), std::get<1>(masks), std::get<2>(masks), std::get<3>(masks)));
//		rgb_surface2_.reset(SDL_CreateRGBSurface(SDL_SWSURFACE, resolution_.width, resolution_.height, bpp,
//				std::get<0>(masks), std::get<1>(masks), std::get<2>(masks), std::get<3>(masks)),
//				[](SDL_Surface*s){SDL_FreeSurface(s);});
	}
	if (bpp != sdl_bpp_) {
		sdl_bpp_ = bpp;
		sdl_resize(resolution_);
	}
	return true;
}
} /* namespace sdl_window */
} /* namespace yuri */
