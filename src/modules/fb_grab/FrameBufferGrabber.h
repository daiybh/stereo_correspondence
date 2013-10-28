/*
 * FrameBufferGrabber.h
 *
 *  Created on: Oct 5, 2010
 *      Author: worker
 */

#ifndef FRAMEBUFFERGRABBER_H_
#define FRAMEBUFFERGRABBER_H_

#include "yuri/core/IOThread.h"
extern "C" {
#include "GL/glx.h"
}
#include "yuri/graphics/GL.h"

#ifndef YURI_ANDROID
//#include <boost/date_time/posix_time/posix_time.hpp>
#endif
namespace yuri {

namespace fbgrab {

//using namespace boost::posix_time;
//using namespace yuri::graphics;
struct _context_info {
	uint index;
	GLint x;
	GLint y;
	GLsizei width;
	GLsizei height;
	yuri::size_t frame_size;
	GLenum format;
	bool stereo;
	bool clean;
	long update_method;
	GLint alignment;
	bool local;
	bool flip_y;
};

#define 	YURI_GRAB_NONE		0
#define 	YURI_GRAB_DIRECT	1
#define 	YURI_GRAB_INDIRECT	2
#define 	YURI_GRAB_MAX		2

#define 	YURI_GRAB_ACTUAL	0x100
#define 	YURI_GRAB_FRAME		0x101
#define 	YURI_GRAB_ABSOLUTE	0x102

class FrameBufferGrabber: public core::IOThread {
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();

	FrameBufferGrabber(log::Log &_log, core::pwThreadBase parent, core::Parameters& parameters);
	virtual ~FrameBufferGrabber();
	void *get_function(std::string func);
	void set_viewport(GLint x, GLint y, GLsizei width, GLsizei height);
	void set_method(long method0);
	void set_method(std::string method0);
	void set_default_format(std::string format);
	void set_update_method(std::string format);
	void pre_swap();
	void post_swap();
	bool step() {sleep(latency);return true;}
	bool add_grabbing_context(GLXContext ctx);
	bool set_param(const core::Parameter& parameter);
protected:
	std::vector<void*> handles;
	yuri::mutex resolve_lock;
	std::map<std::string,void*> functions;
	std::map<GLXContext,_context_info> contexts;
	long method, update_method;
	GLenum default_format;
	bool flip_y, show_avatar;
	graphics::GL gl;
	core::pBasicFrame in_frame;
	time_value grab_start;
	size_t measure;
	time_duration accumulated_time;
	size_t measurement_frames;
	void pre_swap_direct();
	void pre_swap_indirect();
	void post_swap_indirect();
	_context_info &get_current_context_info();
	static yuri::size_t get_format_bytes_per_pixel(GLenum format);
	static long gl_to_yuri_format(long format);
	inline bool check_gl_error();
	static inline void restore_gl(GLint read, GLint draw, bool restore_client);
	static void update_context(_context_info& ctx, GLint x, GLint y, GLsizei width, GLsizei height);
	static void flip_memory(_context_info& ctx, core::pBasicFrame& data);
	void draw_avatar();
	bool do_add_grabbing_context(GLXContext ctx);
	//static std::map<yuri::format_t, GLenum> yuri_gl_formats;

};

}

}

#endif /* FRAMEBUFFERGRABBER_H_ */
