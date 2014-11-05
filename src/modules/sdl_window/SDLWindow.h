/*!
 * @file 		SDLWindow.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		24.09.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SDLWINDOW_H_
#define SDLWINDOW_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "SDL.h"
#ifdef  YURI_SDL_OPENGL
#include "yuri/gl/GL.h"
#endif

namespace yuri {
namespace sdl_window {

class SDLWindow: public core::SpecializedIOFilter<core::RawVideoFrame>
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	SDLWindow(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~SDLWindow() noexcept;
private:
	
	virtual void run() override;
	virtual bool step() override;
	virtual core::pFrame do_special_single_step(const core::pRawVideoFrame& frame) override;
	virtual bool set_param(const core::Parameter& param);
	void process_sdl_events();
	void sdl_resize(resolution_t);
	bool prepare_rgb_overlay(const core::pRawVideoFrame& frame);
	resolution_t	resolution_;
	bool			fullscreen_;
	bool			default_keys_;
	bool			use_gl_;
	SDL_Surface*	surface_;
	unique_ptr<SDL_Overlay, std::function<void(SDL_Overlay*)>>	overlay_;
	unique_ptr<SDL_Surface, std::function<void(SDL_Surface*)>>	rgb_surface_;
	int				sdl_bpp_;
	std::string		title_;
#ifdef  YURI_SDL_OPENGL
	gl::GL				gl_;
	std::string			transform_shader_;
	std::string			color_map_shader_;
	bool				flip_x_;
	bool				flip_y_;
	bool				read_back_;
	int					shader_version_;
#endif
};

} /* namespace sdl_window */
} /* namespace yuri */
#endif /* SDLWINDOW_H_ */
