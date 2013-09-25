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
	
	virtual bool step();
	virtual bool set_param(const core::Parameter& param);
	void process_sdl_events();
	resolution_t	resolution_;
	bool			fullscreen_;
	shared_ptr<SDL_Overlay>	overlay_;
	SDL_Surface*	surface_;
};

} /* namespace sdl_window */
} /* namespace yuri */
#endif /* SDLWINDOW_H_ */
