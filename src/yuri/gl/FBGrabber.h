/*
 * FBGrabber.h
 *
 *  Created on: 24. 1. 2015
 *      Author: neneko
 */

#ifndef FBWRAPPER_H_
#define FBWRAPPER_H_

#include "yuri/gl/GL.h"
#include "yuri/core/thread/IOThread.h"

#include <dlfcn.h>
extern "C" {
	#include "GL/glx.h"
}

namespace yuri{
namespace fb_grabber {


class FBGrabber: public core::IOThread {
public:
	FBGrabber(log::Log& log_, core::pwThreadBase parent, const core::Parameters& params);
	virtual ~FBGrabber() noexcept;

	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();

	void* get_func(const std::string& name);
	void set_viewport(geometry_t geometry);
	void pre_swap();
	void post_swap();

private:
	struct context_info_t {
		context_info_t(GLXContext context, position_t index, geometry_t geometry = {}):
			context(context),index(index),geometry(geometry) {}
		context_info_t(const context_info_t&) = delete;
		context_info_t(context_info_t && rhs) noexcept :
			context(rhs.context), index(rhs.index),geometry(rhs.geometry)
		{
		}
		context_info_t& operator=(const context_info_t&) = delete;
		~context_info_t() noexcept {}
		GLXContext context;
		position_t index;
		geometry_t geometry;
	};

	enum class grab_type_t {
		pre_swap,
		post_swap
	};
	virtual bool step() override;
	virtual	void do_connect_in(position_t position, core::pPipe pipe) override;
	virtual	void do_connect_out(position_t position, core::pPipe pipe) override;

	context_info_t& get_context();
	virtual bool set_param(const core::Parameter &parameter) override;
	void read_buffers(const context_info_t& ctx, GLenum buffer, int idx);

	gl::GL gl_;
	void* gl_handle_;
	void *sdl_handle_;
	std::string gl_lib_path_;
	std::string sdl_lib_path_;

	format_t format_;
	bool stereo_;
	bool depth_;
	int step_;

	grab_type_t grab_type_;


	std::mutex context_mutex_;
	std::map<GLXContext, context_info_t> contexts_;
	std::atomic<position_t> next_index_;
};

}
}



#endif /* FBWRAPPER_H_ */
