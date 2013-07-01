/*!
 * @file 		SimpleRenderer.h
 * @author 		Zdenek Travnicek
 * @date 		11.8.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef SIMPLERENDERER_H_
#define SIMPLERENDERER_H_

#include "yuri/core/BasicIOThread.h"
#ifdef YURI_HAVE_GTKMM
#include "yuri/graphics/GTKWindow.h"
#else
#include "yuri/graphics/GLXWindow.h"
#endif
#include "yuri/graphics/GL.h"
//#include <boost/thread/mutex.hpp>
namespace yuri {

namespace renderer {
//using boost::dynamic_pointer_cast;
//using boost::posix_time::time_duration;
//using boost::posix_time::ptime;
namespace stereo {
	enum _type {
		none,
		anaglyph,
		quadbuffer
	};
}
class SimpleRenderer: public yuri::core::BasicIOThread {
public:
	SimpleRenderer(log::Log &log_,core::pwThreadBase parent, core::Parameters &p);
	virtual ~SimpleRenderer();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters configure();
	void run();
	bool step();
	void set_keep_aspect(bool a);
	bool set_param(const core::Parameter &parameter);
protected:
	core::pCallback drawcb, initcb;
	std::vector<core::pBasicFrame > frames;
	mutex draw_lock;
	//GLuint tid;
	yuri::size_t c;
	//float tx, ty;
	std::vector<bool> changed;
	bool keep_aspect;
	bool flip_x, flip_y, flip_x_right, flip_y_right;
	yuri::ubyte_t quality;
	bool vsync;

	static void init_gl(core::pwThreadBase global, core::pwThreadBase data);
	static void draw_gl(core::pwThreadBase global, core::pwThreadBase data);
	void _draw_gl(shared_ptr<graphics::WindowBase> win);
	void generate_texture(yuri::uint_t index = 0);
	bool prepare_image(yuri::uint_t index = 0);

	graphics::GL gl;
	size_t measure;
	time_duration accumulated_time;
	size_t measurement_frames;
	bool stereo_3d;
	stereo::_type stereo_type;
	float stereo_correction;
	bool swap_eyes;
};

}

}

#endif /* SIMPLERENDERER_H_ */

