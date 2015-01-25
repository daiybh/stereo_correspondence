/*!
 * @file 		GlxWindow.h
 * @author 		<Your name>
 * @date 		25.01.2015
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#ifndef GLXWINDOW_H_
#define GLXWINDOW_H_

#include "yuri/core/thread/IOThread.h"
#include "GL/glx.h"
#include "yuri/gl/GL.h"
namespace yuri {
namespace glx_window {

class GlxWindow: public core::IOThread
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	GlxWindow(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~GlxWindow() noexcept;
private:
	virtual void run() override;
	virtual bool step() override;
	virtual bool set_param(const core::Parameter& param);
	bool create_window();
	bool create_glx_context();
	bool show_window(bool show = true);
	bool show_cursor(bool show = false);
	void move_window(coordinates_t coord);
	void resize_window(resolution_t res);

	bool process_x11_events();
	bool resize_event(geometry_t geometry);
	bool swap_buffers();
private:
	gl::GL					gl_;
	using display_deleter = std::function<void(Display*)>;
	std::string 			screen_;
	unique_ptr<Display, display_deleter>
							display_;
	Window					root_;
	Window					win_;
	int						screen_number_;
	std::vector<GLint>		attributes_;
	geometry_t				geometry_;
	XVisualInfo*			visual_;
	GLXContext				glx_context_;

	bool					flip_x_;
	bool					flip_y_;
	bool					read_back_;
};

} /* namespace glx_window */
} /* namespace yuri */
#endif /* GLXWINDOW_H_ */

