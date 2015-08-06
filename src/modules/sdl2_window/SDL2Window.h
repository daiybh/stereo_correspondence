/*!
 * @file 		SDL2Window.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		06.08.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SDL2WINDOW_H_
#define SDL2WINDOW_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/core/thread/Convert.h"
#include <SDL.h>
namespace yuri {
namespace sdl2_window {

class SDL2Window: public core::IOThread
{
	using base_type = core::IOThread;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters	configure();
							SDL2Window(const log::Log &log_,
			core::pwThreadBase parent, const core::Parameters &parameters);
	virtual 				~SDL2Window() noexcept;
private:
	virtual void 			run() override;
	virtual bool 			set_param(const core::Parameter& param) override;
	bool 					process_sdl_events();
	bool 					create_renderer();
	bool					verify_texture_format(const core::pRawVideoFrame& frame);

	resolution_t 			window_resolution_;
	coordinates_t 			window_position_;

	std::string				window_title_;
	bool					window_decorations_;
	bool					window_resizable_;
	bool					window_fullscreen_;
	bool					window_keep_aspect_;

	int						screen_number_;

	std::unique_ptr<SDL_Window, std::function<void(SDL_Window*)>>
							window_;
	std::unique_ptr<SDL_Renderer, std::function<void(SDL_Renderer*)>>
							renderer_;
	std::unique_ptr<SDL_Texture, std::function<void(SDL_Texture*)>>
							texture_;

	format_t 				last_texture_format_;
	resolution_t 			last_texture_res_;


	std::shared_ptr<core::Convert>
							converter_;
	//SDL_Surface* screen_;
	std::vector<format_t>	supported_formats_;
};

} /* namespace sdl2_window */
} /* namespace yuri */
#endif /* SDL2WINDOW_H_ */
