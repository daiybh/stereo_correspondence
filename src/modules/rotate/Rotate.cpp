/*
 * Rotate.cpp
 *
 *  Created on: 9.4.2013
 *      Author: neneko
 */


#include "Rotate.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace rotate {

REGISTER("rotate",Rotate)
IO_THREAD_GENERATOR(Rotate)

core::pParameters Rotate::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("Rotate module.");
	(*p)["angle"]["Angle in degrees CW to rotate (supported values are 0, 90, 180, 270)"]=90;
	p->set_max_pipes(1,1);
	return p;
}


Rotate::Rotate(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,1,1,std::string("rotate")),angle_(90)
{
	IO_THREAD_INIT("Rotate")
}

Rotate::~Rotate()
{
}

namespace {
core::pBasicFrame rotate(const core::pBasicFrame& frame, size_t angle) {
	if (!frame) return core::pBasicFrame();
	const size_t width = frame->get_width();
	const size_t height = frame->get_height();
	core::pBasicFrame output;
	if (angle == 90 || angle==270) output = core::BasicIOThread::allocate_empty_frame(YURI_FMT_RGB, height, width, true);
	else if (angle == 180) output = core::BasicIOThread::allocate_empty_frame(YURI_FMT_RGB, width, height, true);
	else return core::pBasicFrame();
	const ubyte_t * src = PLANE_RAW_DATA(frame,0);
	ubyte_t * dest = PLANE_RAW_DATA(output,0);
	if (angle == 90) {
		for (size_t y = 0; y < height; ++y) {
			for (size_t x = 0; x < width; ++x) {
				const size_t new_x = height-y-1;
				const size_t new_y = x;
				const size_t pos = 3*(new_y*height+new_x);
				if(pos>=PLANE_SIZE(output,0)) {
					std::cerr<<"Pos out of range! " << pos << " instead of " << PLANE_SIZE(output,0) <<
							"x: " << x << ", y: " << y << ", new_x: " << new_x << ", new_y: " << new_y <<"\n";
				}
				std::copy(src,src+3,&dest[pos]);
				src+=3;
			}
		}
	} else if (angle == 270) {
		for (size_t y = 0; y < height; ++y) {
			for (size_t x = 0; x < width; ++x) {
				const size_t new_x = y;
				const size_t new_y = width-x-1;
				const size_t pos = 3*(new_y*height+new_x);
				std::copy(src,src+3,&dest[pos]);
				src+=3;
			}
		}
	} else if (angle == 180) {
		for (size_t y = 0; y < height; ++y) {
			for (size_t x = 0; x < width; ++x) {
				const size_t new_x = width - x - 1;
				const size_t new_y = height - y - 1;
				const size_t pos = 3*(new_y*width+new_x);
				std::copy(src,src+3,&dest[pos]);
				src+=3;
			}
		}
	}
	return output;
}
}

bool Rotate::step()
{
	if (!in[0]) return true;
	core::pBasicFrame frame = in[0]->pop_frame();
	if (!frame) return true;

	if(!angle_) push_raw_frame(0,frame);
	else {
		if (frame->get_format() != YURI_FMT_RGB) {
			log[log::warning] << "Currently only 24bit RGB is supported.";
		} else {
			core::pBasicFrame output = rotate(frame, angle_);
			if (output) push_raw_video_frame(0, output);
		}
	}
	return true;
}
bool Rotate::set_param(const core::Parameter &param)
{
	if (param.name == "angle") {
		angle_ = param.get<size_t>();
		if (angle_ != 90 && angle_!=180 && angle_!=270) angle_ = 0;
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace rotate */
} /* namespace yuri */

