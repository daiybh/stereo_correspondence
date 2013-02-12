/*
 * SimpleRenderer.h
 *
 *  Created on: Aug 11, 2010
 *      Author: neneko
 */

#ifndef SIMPLERENDERER_H_
#define SIMPLERENDERER_H_

#include "yuri/io/BasicIOThread.h"
#ifdef YURI_HAVE_GTKMM
#include "yuri/graphics/GTKWindow.h"
#else
#include "yuri/graphics/GLXWindow.h"
#endif
#include "yuri/config/RegisteredClass.h"
#include "yuri/graphics/GL.h"
#include <boost/thread/mutex.hpp>
namespace yuri {

namespace io {
using namespace yuri::config;
using namespace yuri::graphics;
using boost::dynamic_pointer_cast;
using boost::mutex;
using boost::posix_time::time_duration;
using boost::posix_time::ptime;
namespace stereo {
	enum _type {
		none,
		anaglyph,
		quadbuffer
	};
}
class SimpleRenderer: public yuri::io::BasicIOThread {
public:
	SimpleRenderer(Log &log_,pThreadBase parent,Parameters &p);
	virtual ~SimpleRenderer();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();
	void run();
	bool step();
	void set_keep_aspect(bool a);
	bool set_param(Parameter &parameter);
protected:
	shared_ptr<Callback> drawcb, initcb;
	std::vector<pBasicFrame > frames;
	mutex draw_lock;
	//GLuint tid;
	yuri::size_t c;
	//float tx, ty;
	std::vector<bool> changed;
	bool keep_aspect;
	bool flip_x, flip_y, flip_x_right, flip_y_right;
	yuri::ubyte_t quality;
	bool vsync;

	static void init_gl(pThreadBase global, pThreadBase data);
	static void draw_gl(pThreadBase global, pThreadBase data);
	void _draw_gl(shared_ptr<WindowBase> win);
	void generate_texture(yuri::uint_t index = 0);
	bool prepare_image(yuri::uint_t index = 0);

	GL gl;
	size_t measure;
	time_duration accumulated_time;
	size_t measurement_frames;
	bool stereo_3d;
	stereo::_type stereo_type;
	float stereo_correction;
	bool swap_eyes;
	static std::map<std::string,stereo::_type> stereo_types;
};

}

}

#endif /* SIMPLERENDERER_H_ */

