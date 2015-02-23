/*
 * FBGrabber.cpp
 *
 *  Created on: 24. 1. 2015
 *      Author: neneko
 */

#include "FBGrabber.h"
#include "yuri/core/Module.h"

#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#include <array>
#include "yuri/core/frame/raw_frame_params.h"
namespace yuri{
namespace fb_grabber {

namespace {
geometry_t get_current_resolution()
{
	std::array<int,4> dims;
	glGetIntegerv(GL_VIEWPORT, &dims[0]);
	return {static_cast<dimension_t>(dims[2]),
			static_cast<dimension_t>(dims[3]),
			dims[0],
			dims[1]};

	//	auto dpy = glXGetCurrentDisplay();
//	auto drawable = glXGetCurrentDrawable();
//	XWindowAttributes attribs;
//	XGetWindowAttributes(dpy, drawable, &attribs);
//	return {static_cast<dimension_t>(attribs.width),
//			static_cast<dimension_t>(attribs.height),
//			attribs.x,
//			attribs.y};
}
}


MODULE_REGISTRATION_BEGIN("fb_grabber")
		REGISTER_IOTHREAD("fb_grabber",FBGrabber)
MODULE_REGISTRATION_END()

IOTHREAD_GENERATOR(FBGrabber)

core::Parameters FBGrabber::configure()
{
	core::Parameters p  = IOThread::configure();
	p.set_description("Grabs content of the framebuffer");
	p["gl_library"]["Path to libGL.so"]="/usr/lib/libGL.so";
	p["sdl_library"]["Path to libSDL.so"]="/usr/lib/libSDL.so";
	p["format"]["Capture format"]="RGB";
	p["stereo"]["Capture stereo"]=false;
	p["depth"]["Capture depth"]=false;
	return p;
}

FBGrabber::FBGrabber(log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters)
	:IOThread(log_, parent, 0,0,"wrap_builder"),gl_(log),
	 gl_handle_(nullptr),sdl_handle_(nullptr),gl_lib_path_("/usr/lib/libGL.so"),
	 sdl_lib_path_("/usr/lib/libSDL.so"),format_(core::raw_format::rgb24),
	 stereo_(false),depth_(false),step_(1),grab_type_(grab_type_t::pre_swap),
	 next_index_(0)
{

	IOTHREAD_INIT(parameters)
	step_=(1+(stereo_?1:0))*(depth_?2:1);
	if (!gl_lib_path_.empty()) {
		gl_handle_ = dlopen(gl_lib_path_.c_str(), RTLD_LAZY);
		if (!gl_handle_) {
			log[log::error] << "Failed to open libGL.so";
		} else {
			log[log::info] << "libGL.so opened";
		}
	}
	if (!sdl_lib_path_.empty()) {
		sdl_handle_ = dlopen(sdl_lib_path_.c_str(), RTLD_LAZY);
		if (!sdl_handle_) {
			log[log::error] << "Failed to open libSDL.so";
		} else {
			log[log::info] << "libSDL.so opened";
		}
	}



}
FBGrabber::~FBGrabber() noexcept
{
	if (gl_handle_) {
		dlclose(gl_handle_);
	}
	if (sdl_handle_) {
		dlclose(sdl_handle_);
	}
}

bool FBGrabber::step()
{
	return true;
}

void* FBGrabber::get_func(const std::string& name) {
	if (gl_handle_) {
		if (auto handle = dlsym(gl_handle_, name.c_str())) {
			return handle;
		}
	}
	if (sdl_handle_) {
		if (auto handle = dlsym(sdl_handle_, name.c_str())) {
			return handle;
		}
	}
	return nullptr;
}

FBGrabber::context_info_t& FBGrabber::get_context()
{
	GLXContext context = glXGetCurrentContext();
	lock_t _(context_mutex_);
	auto it = contexts_.find(context);
	if (it == contexts_.end()) {
		contexts_.insert(std::make_pair(context, context_info_t{context, next_index_.fetch_add(1)}));
		resize(-1, next_index_);
		return contexts_.at(context);
	}
	return it->second;

}

void FBGrabber::set_viewport(geometry_t geometry)
{
	auto& ctx = get_context();
	ctx.geometry = geometry;
	log[log::info] << "Setting geometry for context " << ctx.context << " to " << geometry;
}


void FBGrabber::read_buffers(const context_info_t& ctx, GLenum buffer, int idx)
{
	GLint act_buffer, pix_buf;
	glGetIntegerv(GL_READ_BUFFER, &act_buffer);
	glGetIntegerv(GL_PIXEL_PACK_BUFFER, &pix_buf);
	glReadBuffer(buffer);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	const auto& frame = gl_.read_window(ctx.geometry, format_);
	push_frame(idx, frame);
	if (depth_) {
		const auto& framed = gl_.read_window(ctx.geometry, core::raw_format::depth8);
		push_frame(idx+1, framed);
	}
	glReadBuffer(static_cast<GLenum>(act_buffer));
	glBindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLenum>(pix_buf));
}
void FBGrabber::pre_swap()
{
	if (grab_type_ == grab_type_t::pre_swap) {
		auto& ctx = get_context();
		if (ctx.geometry.get_resolution() == resolution_t{0,0}) {
			log[log::warning] << "swap called before set_viewport. Approximating the resolution (may be incorrect)";
			ctx.geometry = get_current_resolution();
		}
		read_buffers(ctx, GL_BACK_LEFT, step_ * ctx.index);
		if (stereo_) {
			read_buffers(ctx, GL_BACK_RIGHT, step_ * ctx.index + (depth_?2:1));
		}
	}
}
void FBGrabber::post_swap()
{
	if (grab_type_ == grab_type_t::post_swap) {
		auto& ctx = get_context();
		if (ctx.geometry.get_resolution() == resolution_t{0,0}) {
			log[log::warning] << "swap called before set_viewport. Approximating the resolution (may be incorrect)";
			ctx.geometry = get_current_resolution();
		}
		read_buffers(ctx, GL_FRONT_LEFT, step_ * ctx.index);
		if (stereo_) {
			read_buffers(ctx, GL_FRONT_RIGHT, step_ * ctx.index + (depth_?2:1));
		}
	}

}

void FBGrabber::do_connect_in(position_t position, core::pPipe pipe)
{
	if (position >= do_get_no_in_ports()) {
		resize(position+1, -1);
	}
	return IOThread::do_connect_in(position, pipe);
}
void FBGrabber::do_connect_out(position_t position, core::pPipe pipe)
{
	if (position >= do_get_no_out_ports()) {
		resize(-1, position+1);
	}
	return IOThread::do_connect_out(position, pipe);
}

bool FBGrabber::set_param(const core::Parameter &parameter)
{
	if (assign_parameters(parameter)
			(gl_lib_path_, "gl_library")
			(sdl_lib_path_, "sdl_library")
			.parsed<std::string>
				(format_, "format", core::raw_format::parse_format)
			(stereo_, "stereo")
			(depth_, "depth")) {

		if (!format_) {
			log[log::warning] << "Failed to parse format, using RGB";
			format_ = core::raw_format::rgb24;
		}
		return true;
		}
	return IOThread::set_param(parameter);
}
}
}
