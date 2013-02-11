/*
 * SimpleRenderer.cpp
 *
 *  Created on: Aug 11, 2010
 *      Author: neneko
 */

#include "SimpleRenderer.h"
#include <boost/assign.hpp>
#include <boost/algorithm/string.hpp>
namespace yuri {

namespace io {
using boost::posix_time::microsec_clock;

REGISTER("simple_renderer",SimpleRenderer)

shared_ptr<BasicIOThread> SimpleRenderer::generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception)
{
	shared_ptr<SimpleRenderer> sr (new SimpleRenderer(_log,parent,parameters));
	return sr;
}
shared_ptr<Parameters> SimpleRenderer::configure()
{
	shared_ptr<Parameters> p =
#ifdef YURI_HAVE_GTKMM
			GTKWindow::configure();
#else
			GLXWindow::configure();
#endif
	(*p)["keep_aspect"]["Keep aspect ratio"]=false;
	(*p)["flip_x"]["Flip around X axis"]=false;
	(*p)["flip_y"]["Flip around Y axis"]=false;
	(*p)["vsync"]["Synchronize to vsync"]=false;
	(*p)["quality"]["Rendering quality for texture types that have multiple quality settings. Valid range varies, 0 should always be best quality, 1 safe"]=1;
	(*p)["measure"]["Number of samples to measure read-back time. ) to disable"]=0;
	(*p)["window_type"]["Type of underlying window. GLX, GTK."]=string("GLX");
	(*p)["stereo_type"]["Type of stereoscopic output (none, anaglyph, quadbuffer)"]=string("none");
	(*p)["correction"]["Amount by which the stereoscopic pair should be corrected."]=0.0;
	(*p)["flip_y_right"]["Flip right image around Y axis"]=false;
	(*p)["flip_x_right"]["Flip right image around X axis"]=false;
	(*p)["swap_eyes"]["Swap stereoscopic eyes"]=false;
	p->set_max_pipes(1,0);
	p->add_input_format(YURI_FMT_RGB);
	p->add_input_format(YURI_FMT_RGBA);
	p->add_input_format(YURI_FMT_YUV422);
	return p;
}

map<string,stereo::_type> SimpleRenderer::stereo_types = boost::assign::map_list_of<string,stereo::_type>
	("none",stereo::none)
	("anaglyph",stereo::anaglyph)
	("quadbuffer",stereo::quadbuffer);

SimpleRenderer::SimpleRenderer(Log &log_,pThreadBase parent,Parameters &parameters)
	:BasicIOThread(log_,parent,1,0,"SimpleRenderer"),keep_aspect(false),
	 flip_x(false),flip_y(false),flip_y_right(false),quality(1),vsync(false),gl(log),
	 measure(0),measurement_frames(0),stereo_3d(false),stereo_type(stereo::none),
	 stereo_correction(0.0),swap_eyes(false)
{
	IO_THREAD_INIT("SimpleRenderer")
	if (stereo_3d) resize(2,0);
	gl.textures[0].keep_aspect = keep_aspect;
	gl.textures[0].flip_x = flip_x;
	gl.textures[0].flip_y = flip_y;

	if (stereo_3d) {
		gl.textures[1].keep_aspect = keep_aspect;
		gl.textures[1].flip_x = flip_x_right;
		gl.textures[1].flip_y = flip_y_right;
		frames.resize(2);
		changed.resize(2,true);
	} else {
		frames.resize(1);
		changed.resize(1,true);
	}
	latency=10000;
}

SimpleRenderer::~SimpleRenderer()
{

}

void SimpleRenderer::run()
{
	IO_THREAD_PRE_RUN
	set_thread_name(string("Renderer"));
	drawcb.reset(new Callback(SimpleRenderer::draw_gl,get_this_ptr()));
	initcb.reset(new Callback(SimpleRenderer::init_gl,get_this_ptr()));
	params["draw_callback"]=drawcb;
	params["init_callback"]=initcb;
	shared_ptr<GLXWindow> glx;
#ifdef YURI_HAVE_GTKMM
	if (boost::iequals(params["window_type"].get<std::string>(),"gtk")) {
		glx.reset(new GTKWindow(log,get_this_ptr(),params));
	} else
#endif
	glx.reset(new GLXWindow(log,get_this_ptr(),params));


	if (!glx->create()) {
		log[error] << "Failed to create window" << endl;
		throw Exception("Failed to create GLX window");
	}
	glx->show();
	glx->set_vsync(vsync);
	glx->exec(initcb);
	while (still_running()) {
		while (glx->process_events()) {
			if (glx->check_key(9)) {
				exitCode = YURI_EXIT_USER_BREAK;
				request_end();
			}
			break;
		}
		if (!step()) break;
		glx->exec(drawcb);
		ThreadBase::sleep(latency);
		glx->swap_buffers();
		if (vsync) gl.finish_frame();
	}
	IO_THREAD_POST_RUN
}

void SimpleRenderer::init_gl(pThreadBase global, pThreadBase /*data*/)
{
	if (global.expired()) return;
	shared_ptr<WindowBase> win = dynamic_pointer_cast<WindowBase>(global.lock());

	GL::enable_smoothing();

}

void SimpleRenderer::draw_gl(pThreadBase global, pThreadBase data)
{
	if (global.expired()) throw("global expired");
	//shared_ptr<ThreadBase> tb = data.lock();
	shared_ptr<SimpleRenderer> simple = dynamic_pointer_cast<SimpleRenderer>(data.lock());
	if (data.expired()) throw("data expired");
	shared_ptr<WindowBase> win = dynamic_pointer_cast<WindowBase>(global.lock());
	assert (simple);
	assert(win);
	simple->_draw_gl(win);
}

void SimpleRenderer::_draw_gl(shared_ptr<WindowBase> win)
{
	if (!prepare_image(0)) return;
	if (stereo_3d && !prepare_image(1)) return;
	glDrawBuffer(GL_BACK_LEFT);
	gl.setup_ortho();
	if (!stereo_3d) {
		gl.draw_texture(0,win);
	} else {
		yuri::uint_t left = 0, right = 1;
		if (swap_eyes) {
			left=1;right=0;
		}
		switch(stereo_type) {
			case stereo::anaglyph: glColorMask(1,0,0,1);break;
			case stereo::quadbuffer: glDrawBuffer(GL_BACK_LEFT); break;
			default:break;
		}
		gl.textures[left].dx = gl.textures[left].tx*stereo_correction/200.0;
		gl.draw_texture(left,win);
		switch(stereo_type) {
			case stereo::anaglyph: glColorMask(0,1,1,1);break;
			case stereo::quadbuffer: glDrawBuffer(GL_BACK_RIGHT); break;
			default:break;
		}
		gl.textures[right].dx = -gl.textures[right].tx*stereo_correction/200.0;
		gl.draw_texture(right,win);
	}

}

bool SimpleRenderer::step()
{
	if (in[0] && !in[0]->is_empty()) {
		while (!in[0]->is_empty()) {
			mutex::scoped_lock l(draw_lock);
			frames[0] = in[0]->pop_frame();
		}
		assert(frames[0]);
		changed[0]=true;
	}
	if (stereo_3d && in[1] && !in[1]->is_empty()) {
		while (!in[1]->is_empty()) {
			mutex::scoped_lock l(draw_lock);
			frames[1] = in[1]->pop_frame();
		}
		assert(frames[1]);
		changed[1]=true;
	}
	return true;
}

void SimpleRenderer::generate_texture(yuri::uint_t index)
{
	mutex::scoped_lock l(draw_lock);
	if (changed[index]) {
		ptime _start_time;
		if (measure) _start_time = microsec_clock::local_time();
		gl.generate_texture(index,frames[index]);
		if (measure) {
			ptime _end_time=microsec_clock::local_time();
			if (measurement_frames >= measure) {
				log[info] << "Generating "<< measurement_frames<< " textures took " << to_simple_string(accumulated_time) << " that is " << (accumulated_time/measurement_frames).total_microseconds() << " us per frame"  << endl;
				accumulated_time = boost::posix_time::microseconds(0);
				measurement_frames=0;
			}
			accumulated_time += (_end_time - _start_time);
			measurement_frames++;
		}
	}
	changed[index] = false;
}
bool SimpleRenderer::prepare_image(yuri::uint_t index)
{
	if (!frames[index]) {
		log[verbose_debug] << "I have empty frame, skipping" << endl;
		return false;
	}
	switch (frames[index]->get_format()) {
		case YURI_FMT_RGB:
		case YURI_FMT_RGBA:
		case YURI_FMT_BGR:
		case YURI_FMT_BGRA:
		case YURI_FMT_YUV422:
		case YURI_FMT_YUV444:
		case YURI_FMT_YUV420_PLANAR:
		case YURI_FMT_YUV422_PLANAR:
		case YURI_FMT_YUV444_PLANAR:
		case YURI_FMT_RED8:
		case YURI_FMT_GREEN8:
		case YURI_FMT_BLUE8:
		case YURI_FMT_Y8:
		case YURI_FMT_U8:
		case YURI_FMT_V8:
		case YURI_FMT_DEPTH8:
		case YURI_FMT_DXT1:
		//case YURI_FMT_DXT2:
		case YURI_FMT_DXT3:
		//case YURI_FMT_DXT4:
		case YURI_FMT_DXT5:
		case YURI_FMT_DXT1_WITH_MIPMAPS:
		//case YURI_FMT_DXT2_WITH_MIPMAPS:
		case YURI_FMT_DXT3_WITH_MIPMAPS:
		//case YURI_FMT_DXT4_WITH_MIPMAPS:
		case YURI_FMT_DXT5_WITH_MIPMAPS:
			break;
		default:
			log[warning] << "Wrong format (" << BasicPipe::get_format_string(frames[index]->get_format()) << ")" << endl;
			return false;
	}

	generate_texture(index);
	return true;
}
void SimpleRenderer::set_keep_aspect(bool a)
{
	keep_aspect = a;
}

bool SimpleRenderer::set_param(Parameter &parameter)
{
	mutex::scoped_lock l(draw_lock);
	if (parameter.name == "keep_aspect")
		keep_aspect=parameter.get<bool>();
	else if (parameter.name == "flip_x")
		flip_x = parameter.get<bool>();
	else if (parameter.name == "flip_y")
		flip_y = parameter.get<bool>();
	else if (parameter.name == "quality") {
		quality = parameter.get<yuri::uint_t>();
		gl.set_lq422(quality);
	} else if (parameter.name == "vsync") {
		vsync = parameter.get<bool>();
	} else if (parameter.name == "measure") {
		measure=parameter.get<yuri::size_t>();
	} else if (parameter.name == "stereo") {
		if (parameter.get<bool>()) {
			stereo_3d = true;
			stereo_type = stereo::quadbuffer;
		}
	} else if (parameter.name == "stereo_type") {
		string stype = parameter.get<string>();
		boost::algorithm::to_lower(stype);
		if (stereo_types.count(stype)) {
			stereo_type = stereo_types[stype];
		} else stereo_type = stereo::none;
		if (stereo_type == stereo::none) {
			stereo_3d = false;
		} else {
			stereo_3d = true;
		}
	} else if (parameter.name == "correction") {
		stereo_correction=parameter.get<float>();
	} else if (parameter.name == "flip_x_right") {
		flip_x_right=parameter.get<bool>();
	} else if (parameter.name == "flip_y_right") {
		flip_y_right=parameter.get<bool>();
	} else if (parameter.name == "swap_eyes") {
		swap_eyes=parameter.get<bool>();
	} else return BasicIOThread::set_param(parameter);
	return true;
}
}

}
