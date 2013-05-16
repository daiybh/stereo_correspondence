/*
 * FrameBufferGrabber.cpp
 *
 *  Created on: Oct 5, 2010
 *      Author: worker
 */

#include "FrameBufferGrabber.h"
#include "yuri/graphics/GLXWindow.h"
#include "yuri/core/Module.h"
#include <dlfcn.h>
#include <GL/glu.h>
#include <boost/assign.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
namespace yuri {

namespace fbgrab {
using yuri::graphics::GLXWindow;
using boost::iequals;
REGISTER("fb_grabber",FrameBufferGrabber)

IO_THREAD_GENERATOR(FrameBufferGrabber)

core::pParameters FrameBufferGrabber::configure()
{
	core::pParameters p = BasicIOThread::configure();
	(*p)["outputs"]["Number of outputs"]=1;
	(*p)["libgl_path"]["Path to libGL.so"]="/usr/lib/libGL.so";
	(*p)["libsdl_path"]["Path to libSDL.so"]="/usr/lib/libSDL.so";
	(*p)["method"]["Method to use (none,direct,indirect)"]="direct";
	(*p)["default_format"]["Default format to use for grabbing (RGB,RGBA)"]="rgb";
	(*p)["update_method"]["Update method for determining area to grab. 'frame' should be the best option for most situations. (actual,frame,absolute)"]="frame";
	(*p)["flip_y"]["Flip image upside down (ie. to normal orientation)"]=true;
	(*p)["avatar"]["Show avatar"]=true;
	(*p)["depth"]["Grab depth buffer"]=false;
	(*p)["measure"]["Number of samples to measure read-back time. ) to disable"]=0;
	return p;
}

std::map<yuri::format_t, GLenum> FrameBufferGrabber::yuri_gl_formats = boost::assign::map_list_of<yuri::format_t, GLenum>
			(YURI_FMT_RGB, GL_RGB)
			(YURI_FMT_RGBA, GL_RGBA)
			(YURI_FMT_BGR, GL_BGR)
			(YURI_FMT_BGRA, GL_BGRA);

FrameBufferGrabber::FrameBufferGrabber(log::Log &_log, core::pwThreadBase parent, core::Parameters& parameters):
		BasicIOThread(_log,parent,1,0,"Frame Buffer Grabber"),
		method(YURI_GRAB_NONE),update_method(YURI_GRAB_ACTUAL),flip_y(true),
		show_avatar(false),gl(log)
{
	IO_THREAD_INIT("FrameBufferGrabber")
	//params.merge(*configure());
	//params.merge(parameters);
	log[log::debug] << "outputs: " << params["outputs"].get<int>();
	resize(in_ports,params["outputs"].get<int>());
	set_method(params["method"].get<std::string>());
	set_default_format(params["default_format"].get<std::string>());
	set_update_method(params["update_method"].get<std::string>());
	flip_y=params["flip_y"].get<bool>();
	show_avatar=params["avatar"].get<bool>();
	void *handle = 0;
	handle = dlopen(params["libgl_path"].get<std::string>().c_str(),RTLD_LAZY);
	if (handle) handles.push_back(handle);
	else {
		log[log::warning] << "Failed to get handle to ligGL.so. ";
	}
	handle = dlopen(params["libsdl_path"].get<std::string>().c_str(),RTLD_LAZY);
	if (handle) handles.push_back(handle);
	else {
		log[log::warning] << "Failed to get handle to ligSDL.so. ";
	}
	if (handles.empty()) {
		log[log::error] << "Neither of the libraries was successfully opened.";
		throw exception::InitializationFailed("No usable libraries");
	}
	grab_start = boost::posix_time::microsec_clock::local_time();
	//putenv("DISPLAY=:0.0");
}

FrameBufferGrabber::~FrameBufferGrabber()
{

}

void *FrameBufferGrabber::get_function(std::string name)
{
	boost::mutex::scoped_lock l(resolve_lock);

	if (functions.find(name) != functions.end())
		return functions[name];
	log[log::info] << "Looking up " << name;
	void* handle=0, *ptr=0;
	BOOST_FOREACH(handle,handles) {
		ptr = dlsym(handle,name.c_str());
		if (ptr) {
			functions[name]=ptr;
			return ptr;
		}
	}
	log[log::error] << "Requested function '" << name << "' not found!";
	return 0;
}

void FrameBufferGrabber::set_viewport(GLint x, GLint y, GLsizei width, GLsizei height)
{
	GLXContext context = glXGetCurrentContext();
	if (!context) return;
	if (contexts.find(context) == contexts.end()) {
		if (GLXWindow::is_context_used(context)) {
			log[log::info] << "Ignoring local context "<< std::hex << context << std::dec;
			contexts[context].local = true;
		} else {
			do_add_grabbing_context(context);
		}
	}
	if (!contexts[context].local) update_context(contexts[context], x, y, width, height);
}

void FrameBufferGrabber::pre_swap()
{
	switch(method) {
		case YURI_GRAB_NONE:
			break;
		case YURI_GRAB_DIRECT:
			pre_swap_direct();
			break;
		case YURI_GRAB_INDIRECT:
			pre_swap_indirect();
			break;
		default:
			break;
	}
	if (show_avatar) draw_avatar();
}
void FrameBufferGrabber::post_swap()
{

	switch(method) {
		case YURI_GRAB_NONE:
			break;
		case YURI_GRAB_DIRECT:
			break;
		case YURI_GRAB_INDIRECT:
			post_swap_indirect();
			break;
		default:
			return;
	}
	return;
}

void FrameBufferGrabber::set_method(long method0)
{
	if (method0<YURI_GRAB_NONE || method0 > YURI_GRAB_MAX) {
		method = YURI_GRAB_NONE;
		return;
	}
	method = method0;
}

void FrameBufferGrabber::set_method(std::string method0)
{
	if (iequals(method0,"direct")) set_method(YURI_GRAB_DIRECT);
	else if (iequals(method0,"indirect")) set_method(YURI_GRAB_INDIRECT);
	else set_method(YURI_GRAB_NONE);
}

void FrameBufferGrabber::pre_swap_direct()
{
//	log[warning] << "Direct grabbing not implemented yet" << endl;
	boost::posix_time::ptime _start_time;
	if (measure) _start_time = boost::posix_time::microsec_clock::local_time();
	GLint orig_read_buffer, orig_draw_buffer;
	try {
		boost::posix_time::time_duration duration = boost::posix_time::microsec_clock::local_time() - grab_start;
		yuri::size_t pts = duration.total_microseconds();
		_context_info &ctx = get_current_context_info();
		if (ctx.local) return;
		ctx.clean=true;

		while (glGetError()!=GL_NO_ERROR);
		glGetIntegerv(GL_READ_BUFFER,&orig_read_buffer);
		glGetIntegerv(GL_DRAW_BUFFER,&orig_draw_buffer);
		glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);
		glPixelStorei(GL_PACK_SWAP_BYTES, GL_FALSE);
		glPixelStorei(GL_PACK_LSB_FIRST, GL_FALSE);
		glPixelStorei(GL_PACK_ROW_LENGTH, ctx.width);
		glPixelStorei(GL_PACK_SKIP_ROWS, 0);
		glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		check_gl_error();

		glReadBuffer(GL_BACK);
		if (!check_gl_error()) return restore_gl(orig_read_buffer,orig_draw_buffer,true);

		format_t fmt = gl_to_yuri_format(ctx.format);
		//shared_array<yuri::ubyte_t> data = allocate_memory_block(ctx.frame_size,true);
		core::pBasicFrame frame = allocate_empty_frame(fmt, ctx.width, ctx.height);
		log[log::verbose_debug] << "Grabbing " <<  ctx.width <<"x"<< ctx.height
				<< "+"<<ctx.x<<ctx.y<<", to buffer of " << ctx.frame_size;
		glReadPixels(ctx.x, ctx.y, ctx.width, ctx.height, ctx.format, GL_UNSIGNED_BYTE, reinterpret_cast<GLubyte*>(PLANE_RAW_DATA(frame,0)));
		if (!check_gl_error()) return restore_gl(orig_read_buffer,orig_draw_buffer,true);

		if (out[ctx.index]) {
			if (ctx.flip_y) flip_memory(ctx,frame);
			//frame=allocate_frame_from_memory(data,ctx.frame_size);
			push_video_frame(ctx.index,frame,fmt,ctx.width,ctx.height,pts,0,pts);
		} else {
		//	log[warning] << "output pipe not connected inp: " << out_ports << ", " << ctx.index << endl;
		}
		if (ctx.stereo) {
			glReadBuffer(GL_BACK_RIGHT);
			if (!check_gl_error()) return restore_gl(orig_read_buffer,orig_draw_buffer,true);
			core::pBasicFrame frame = allocate_empty_frame(fmt, ctx.width, ctx.height);
			//data = allocate_memory_block(ctx.frame_size);
			glReadPixels(ctx.x, ctx.y, ctx.width, ctx.height, ctx.format, GL_UNSIGNED_BYTE, reinterpret_cast<GLubyte*>(PLANE_RAW_DATA(frame,0)));
			if (!check_gl_error()) return restore_gl(orig_read_buffer,orig_draw_buffer,true);
			// TODO: quick hack to handle stereo. WILL break multiple window support
			if (out[ctx.index+1]) {
				if (ctx.flip_y) flip_memory(ctx,frame);
				//frame=allocate_frame_from_memory(data,ctx.frame_size);
				push_video_frame(ctx.index+1,frame,gl_to_yuri_format(ctx.format),ctx.width,ctx.height,pts,0,pts);
			}
		}
		restore_gl(orig_read_buffer,orig_draw_buffer,true);
		check_gl_error();
		if (measure) {
			boost::posix_time::ptime _end_time=boost::posix_time::microsec_clock::local_time();
			if (measurement_frames >= measure) {
				log[log::info] << "Grabbing "<< measurement_frames<< " frames took " << boost::posix_time::to_simple_string(accumulated_time) << " that is " << (accumulated_time/measurement_frames).total_microseconds() << " us per frame";
				accumulated_time = boost::posix_time::microseconds(0);
				measurement_frames=0;
			}
			accumulated_time += (_end_time - _start_time);
			measurement_frames++;
		}
	}
	catch (exception::Exception &e) {
		log[log::debug] << "Unknown context - there probably was no call to glViewport()";
		restore_gl(orig_read_buffer,orig_draw_buffer,true);
		return;
	}
	catch(std::bad_alloc &e) {
		log[log::error] << "Failed to allocate memory: " << e.what();
		restore_gl(orig_read_buffer,orig_draw_buffer,true);
		return;
	}

}

void FrameBufferGrabber::pre_swap_indirect()
{
	//log[warning] << "Direct grabbing not implemented yet" << endl;
}

void FrameBufferGrabber::post_swap_indirect()
{
	log[log::warning] << "Indirect grabbing not implemented yet";
}

_context_info &FrameBufferGrabber::get_current_context_info()
{
	GLXContext context = glXGetCurrentContext();
	if (contexts.find(context) == contexts.end()) {
		throw exception::Exception("Context not found");
	}
	return contexts[context];
}

yuri::size_t FrameBufferGrabber::get_format_bytes_per_pixel(GLenum format)
{
	switch (format) {
		case GL_RGB: return 3;
		case GL_RGBA: return 4;
		default:
			return 0;
	}
}

void FrameBufferGrabber::set_default_format(std::string format)
{
	if (iequals(format,"RGB") || iequals(format,"RGB24")) default_format = GL_RGB;
	else if (iequals(format,"RGBA") || iequals(format,"RGB32")) default_format = GL_RGBA;
	else throw exception::Exception("Unknown format");
}

long FrameBufferGrabber::gl_to_yuri_format(long format)
{
	switch (format) {
		case GL_RGB: return YURI_FMT_RGB;
		case GL_RGBA: return YURI_FMT_RGBA;
		default:
			return 0;
	}

}

bool FrameBufferGrabber::check_gl_error()
{
	int err = glGetError();
	if (err == GL_NO_ERROR) return true;
	log[log::warning] << "OpenGL error: " << gluErrorString(err);
	return false;
}

void FrameBufferGrabber::restore_gl(GLint read, GLint draw, bool restore_client)
{
	glReadBuffer(read);
	glDrawBuffer(draw);
	if (restore_client) glPopClientAttrib();
}

void FrameBufferGrabber::update_context(_context_info& ctx, GLint x, GLint y, GLsizei width, GLsizei height)
{
	if (	 ctx.update_method == YURI_GRAB_ACTUAL 						||
			(ctx.update_method == YURI_GRAB_FRAME 		&& ctx.clean) 	||
			(ctx.update_method == YURI_GRAB_ABSOLUTE 	&& ctx.width == 0)) {
		ctx.x 		= x;
		ctx.y 		= y;
		ctx.width 	= width;
		ctx.height 	= height;
	} else {
		if (ctx.x 		> x) 		ctx.x 		= x;
		if (ctx.y 		> y) 		ctx.y 		= y;
		if (ctx.width 	< width) 	ctx.width 	= width;
		if (ctx.height 	< height) 	ctx.height 	= height;
	}
	/*if (ctx.x%2) ctx.x++;
	if (ctx.y%2) ctx.y++;
	if (ctx.width%2) ctx.width--;
	if (ctx.height%2) ctx.height--;*/

	/*ctx.frame_size = ((ctx.width%ctx.alignment)?ctx.width+1:ctx.width)
			* ((ctx.height%ctx.alignment)?ctx.height+1:ctx.height)
			* get_format_bytes_per_pixel(ctx.format);*/
	ctx.frame_size = ctx.width * ctx.height	* get_format_bytes_per_pixel(ctx.format);
	ctx.clean = false;
}

void FrameBufferGrabber::set_update_method(std::string method)
{
	if (iequals(method,"actual")) update_method=YURI_GRAB_ACTUAL;
	else if (iequals(method,"absolute")) update_method=YURI_GRAB_ABSOLUTE;
	else update_method=YURI_GRAB_FRAME;
}

void FrameBufferGrabber::flip_memory(_context_info& ctx, core::pBasicFrame& frame)
{
	yuri::size_t linesize = ctx.width*get_format_bytes_per_pixel(ctx.format);
	for (int i=0; i < (ctx.height/2);++i) {
		yuri::ubyte_t *ptr1 = PLANE_RAW_DATA(frame,0)+i*linesize;
		yuri::ubyte_t *ptr2 = PLANE_RAW_DATA(frame,0)+((ctx.height-i-1)*linesize);
		for (yuri::size_t j=0;j<linesize;++j) std::swap(*ptr1++,*ptr2++);
	}
}
void FrameBufferGrabber::draw_avatar()
{
	GLXContext context = glXGetCurrentContext();
	if (contexts.find(context) == contexts.end() || contexts[context].local) return;
	if (in_ports && in[0]) if (!in[0]->is_empty()) {
		core::pBasicFrame f  = in[0]->pop_latest();
		if (f) in_frame = f;
	}
	if (!in_frame) return;
	//log[info] << "Drawing " << hex << context << dec << endl;
	graphics::GL::save_state();

	gl.generate_texture(contexts[context].index,in_frame);
	gl.setup_ortho();
	shared_ptr<graphics::WindowBase> win;
	glDrawBuffer(GL_BACK);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glColor3f(1.0f,1.0f,1.0f);
	gl.draw_texture(contexts[context].index,win,0.2,0.2);
	graphics::GL::restore_state();
}

bool FrameBufferGrabber::add_grabbing_context(GLXContext ctx)
{
	return do_add_grabbing_context(ctx);
}
bool FrameBufferGrabber::do_add_grabbing_context(GLXContext context)
{
	uint no_contexts = contexts.size();
	log[log::info] << "Found new context "<< std::hex << context << std::dec << ". Assigning to index " << no_contexts;
	contexts[context].index=no_contexts;
	Display *dpy = glXGetCurrentDisplay();
	int screen, connection;
	connection = XConnectionNumber(dpy);
	glXQueryContext(dpy,context,GLX_SCREEN,&screen);
	/*log[info] << "Setting up viewport " << width << "x" << height << " @ " << x
							<< "x" << y << ". Context: " << hex << context
							<< dec << ". On screen " << DisplayString(dpy)
							<< endl;*/
	glGetBooleanv(GL_STEREO, reinterpret_cast<GLboolean*>(&contexts[context].stereo));
	log[log::info] << "Context has " << (contexts[context].stereo?"enabled":"disabled")  << " stereo support.";
	contexts[context].format = default_format;
	contexts[context].clean = true;
	contexts[context].update_method = update_method;
	contexts[context].local = false;
	contexts[context].flip_y = flip_y;
	glGetIntegerv(GL_PACK_ALIGNMENT,&(contexts[context].alignment));
	log[log::info] << "Using alignment to " << contexts[context].alignment << "Bytes";
	return true;
}
bool FrameBufferGrabber::set_param(const core::Parameter& parameter)
{
	if (parameter.name == "measure") {
		measure=parameter.get<yuri::size_t>();
	} else /*if (parameter.name == "field") {
		field=parameter.get<yuri::ushort_t>();
	} else if (parameter.name == "conversion") {
		string tmp =parameter.get<string>();
		if (conversion_types_map.count(tmp)) conversion=conversion_types_map[tmp];
		else conversion=LINE_INTERLACED;
	} else */return BasicIOThread::set_param(parameter);
	return true;
}
}

}



