/*!
 * @file 		Combine.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Combine.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace combine {

REGISTER("combine",Combine)

IO_THREAD_GENERATOR(Combine)

// So we can write log[info] instead of log[log::info]
using namespace yuri::log;

core::pParameters Combine::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("Combine");
	(*p)["x"]["Width of the grid"]=2;
	(*p)["y"]["Height of the grid"]=2;
	p->set_max_pipes(-1,1);
	return p;
}


Combine::Combine(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,1,1,std::string("combine")),x(2),y(2)
{
	IO_THREAD_INIT("Dummy")
	if (x<1 || y<1) throw exception::InitializationFailed("Wrong size of the grid");
	resize(x*y,1);
	frames.resize(x*y);
}

Combine::~Combine()
{
}

bool Combine::step()
{
	const size_t frames_no = x*y;
	frames.resize(frames_no);
	size_t valid_frames=0;
	for (size_t i=0;i<frames_no;++i) {
		if (!in[i]) return true;
		if (!frames[i]) frames[i]=in[i]->pop_frame();
		if (frames[i]) valid_frames++;
	}
	if (valid_frames < frames_no) return true;
	const format_t format = frames[0]->get_format();
	FormatInfo_t fi = core::BasicPipe::get_format_info(format);
	if (fi->planes > 1) {
		log[log::warning] << "Planar formats not supported";
		return true;
	}
	if (fi->compressed) {
		log[log::warning] << "Compressed formats not supported";
		return true;
	}
	size_t bpp = fi->bpp;
	const size_t width  = frames[0]->get_width();
	const size_t height = frames[0]->get_height();
	for (size_t i=1;i<frames_no;++i) {
		if (frames[i]->get_format() != format) {
			log[log::warning] << "Wrong format for frame in pipe " << i;
			frames[i].reset();
			return true;
		}
		if (frames[i]->get_width()!=width && frames[i]->get_height() != height) {
			log[log::warning] << "Wrong size for frame in pipe " << i;
			frames[i].reset();
			return true;
		}
	}
	core::pBasicFrame output = allocate_empty_frame(format,width*x, height*y);
	ubyte_t* out = PLANE_RAW_DATA(output,0);
	size_t sub_line_width=bpp*width/8;
	size_t line_width = sub_line_width*x;
	size_t idx = 0;
	for (size_t idx_y=0;idx_y<y;++idx_y) {
		for (size_t idx_x=0;idx_x<x;++idx_x) {
			const ubyte_t* raw_src = PLANE_RAW_DATA(frames[idx],0);
			for (size_t line=0;line<height;++line) {
				std::copy(raw_src+line*sub_line_width,
						raw_src+(line+1)*sub_line_width,
						out+(idx_y*height+line)*line_width+idx_x*sub_line_width);
			}
			idx++;
		}
	}
	push_raw_video_frame(0,output);
	for (size_t i=0;i<frames_no;++i) {
		frames[i].reset();
	}
	return true;
}
bool Combine::set_param(const core::Parameter& param)
{
	if (param.name == "x") {
		x = param.get<size_t>();
	} else if (param.name == "y") {
		y = param.get<size_t>();
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
