/*!
 * @file 		BlankGenerator.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.9.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "BlankGenerator.h"
#include "yuri/core/Module.h"
#include "boost/algorithm/string/predicate.hpp"

namespace yuri {

namespace blank {

REGISTER("blank",BlankGenerator)

IO_THREAD_GENERATOR(BlankGenerator)

using boost::iequals;
core::pParameters BlankGenerator::configure()
{
	core::pParameters p (new core::Parameters());
	p->set_max_pipes(0,1);
	p->add_output_format(YURI_FMT_YUV422);
	(*p)["format"]["Format (RGB, YUV422, ...)"]="YUV422";
	(*p)["fps"]["Framerate"]="25";
	(*p)["width"]["Width of the image"]="640";
	(*p)["height"]["Height of the image"]="480";
	return p;
}

BlankGenerator::BlankGenerator(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters) IO_THREAD_CONSTRUCTOR:
		BasicIOThread(log_,parent,0,1,"BlankGenerator"),next_time(not_a_date_time),
		fps(25),width(640),height(480),format(YURI_FMT_YUV422)
{
	IO_THREAD_INIT("Blank generator")
	//latency=4e6/fps;
	latency = 4e5/fps;
	if (format==YURI_FMT_NONE) {
		throw exception::InitializationFailed("None or wrong output format specified");
	}
}

BlankGenerator::~BlankGenerator()
{

}

bool BlankGenerator::set_param(const core::Parameter &p)
{
	if (iequals(p.name,"fps")) {
		fps = p.get<float>();
	} else if (iequals(p.name,"width")) {
		width = p.get<yuri::ushort_t>();
	} else if (iequals(p.name,"height")) {
		height = p.get<yuri::ushort_t>();
	} else if (iequals(p.name,"format")) {
		format = core::BasicPipe::get_format_from_string(p.get<std::string>());
		if (format == YURI_FMT_NONE) {
			log[log::error] << "Failed to parse format std::string!" << std::endl;
			return false;
		}
	} else return BasicIOThread::set_param(p);
	return true;
}
void BlankGenerator::run()
{
	time_duration delta = microseconds(1e6)/fps;
	next_time=microsec_clock::local_time();
	core::pBasicFrame frame;
	while(still_running()) {
		if (microsec_clock::local_time() < next_time) {
			ThreadBase::sleep(latency);
			continue;
		}
		frame = generate_frame();
		if (frame) push_raw_video_frame(0,frame->get_copy());
		next_time+=delta;
	}
}
core::pBasicFrame BlankGenerator::generate_frame()
{
	if (blank_frames.count(format)) return blank_frames[format];
	switch (format) {
		case YURI_FMT_YUV422: blank_frames[format] = generate_frame_yuv422();
			return blank_frames[format];
		case YURI_FMT_RGB: blank_frames[format] = generate_frame_rgb();
			return blank_frames[format];
	}
	log[log::error] << "Wrong format" << std::endl;
	return core::pBasicFrame();
}

core::pBasicFrame BlankGenerator::generate_frame_yuv422()
{
	assert(format==YURI_FMT_YUV422);
	core::pBasicFrame frame = BasicIOThread::allocate_empty_frame(YURI_FMT_YUV422,width,height);
	yuri::ubyte_t *data = PLANE_RAW_DATA(frame,0);
	for (yuri::ushort_t y = 0; y<height; ++y) {
		for (yuri::ushort_t x = 0; x < width; ++x) {
			*data++=0;
			*data++=128;
		}
	}
	return frame;
}

core::pBasicFrame BlankGenerator::generate_frame_rgb()
{
	assert(format==YURI_FMT_RGB24);
	core::pBasicFrame frame = BasicIOThread::allocate_empty_frame(YURI_FMT_RGB,width,height);
	yuri::ubyte_t *data = PLANE_RAW_DATA(frame,0);
	for (yuri::ushort_t y = 0; y<height; ++y) {
		for (yuri::ushort_t x = 0; x < width; ++x) {
			*data++=0;
			*data++=0;
			*data++=0;
		}
	}
	return frame;
}

}

}
