/*!
 * @file 		SDLWindow.h
 * @author 		<Your name>
 * @date 		24.09.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef SDLWINDOW_H_
#define SDLWINDOW_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "SDL.h"
namespace yuri {
namespace sdl_window {

class SDLWindow: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	SDLWindow(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~SDLWindow();
private:
	
	virtual void run() override;
	virtual bool step() override;
	virtual bool set_param(const core::Parameter& param);
	void process_sdl_events();
	void sdl_resize(resolution_t);
	bool prepare_rgb_overlay(const core::pRawVideoFrame& frame);
	resolution_t	resolution_;
	bool			fullscreen_;
	bool			default_keys_;
	bool			use_gl_;
	SDL_Surface*	surface_;
	shared_ptr<SDL_Overlay>	overlay_;
	shared_ptr<SDL_Surface>	rgb_surface_;
	shared_ptr<SDL_Surface>	rgb_surface2_;
	int				sdl_bpp_;
};

} /* namespace sdl_window */
} /* namespace yuri */
#endif /* SDLWINDOW_H_ */
