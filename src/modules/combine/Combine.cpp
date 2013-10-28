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
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
namespace yuri {
namespace combine {

IOTHREAD_GENERATOR(Combine)

MODULE_REGISTRATION_BEGIN("combine")
		REGISTER_IOTHREAD("combine",Combine)
MODULE_REGISTRATION_END()

// So we can write log[info] instead of log[log::info]
using namespace yuri::log;

core::Parameters Combine::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Combine");
	p["x"]["Width of the grid"]=2;
	p["y"]["Height of the grid"]=2;
//	p->set_max_pipes(-1,1);
	return p;
}


Combine::Combine(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::MultiIOFilter(log_,parent,1,1,std::string("combine")),x_(2),y_(2)
{
	IOTHREAD_INIT(parameters)
	if (x_<1 || y_<1) throw exception::InitializationFailed("Wrong size of the grid");
	resize(x_ * y_,1);
//	frames.resize(x_ * y_);
}

Combine::~Combine() noexcept
{
}

std::vector<core::pFrame> Combine::do_single_step(const std::vector<core::pFrame>& framesx)
//bool Combine::step()
{
//	const size_t frames_no = x_*y_;
//	frames.resize(frames_no);
//	size_t valid_frames=0;
//	for (size_t i=0;i<frames_no;++i) {
//		if (!in[i]) return true;
//		if (!frames[i]) frames[i]=in[i]->pop_frame();
//		if (frames[i]) valid_frames++;
//	}
//	if (valid_frames < frames_no) return true;
	const format_t format = framesx[0]->get_format();
	const auto& fi = core::raw_format::get_format_info(format);

//	FormatInfo_t fi = core::BasicPipe::get_format_info(format);
	const size_t frames_no = framesx.size();

	if (fi.planes.size() != 1) {
//	if (fi->planes > 1) {
		log[log::warning] << "Planar formats not supported";
		return {};
	}
//	if (fi->compressed) {
//		log[log::warning] << "Compressed formats not supported";
//		return true;
//	}

	std::vector<core::pRawVideoFrame> frames;
	for (auto& x: framesx) {
		auto f = dynamic_pointer_cast<core::RawVideoFrame>(x);
		if (!f) {
			log[log::warning] << "Received non-raw frame.";
			return {};
		}
		frames.push_back(f);
	}

	size_t bpp = fi.planes[0].bit_depth.first/fi.planes[0].bit_depth.second;
	const resolution_t resolution = frames[0]->get_resolution();
	const size_t width  = frames[0]->get_width();
	const size_t height = frames[0]->get_height();
	for (size_t i=1;i<frames_no;++i) {
		if (frames[i]->get_format() != format) {
			log[log::warning] << "Wrong format for frame in pipe " << i;
			//frames[i].reset();
			return {};
		}
		if (frames[i]->get_resolution() != resolution) {
			log[log::warning] << "Wrong size for frame in pipe " << i;
//			frames[i].reset();
//			return true;
			return {};
		}
	}
	core::pRawVideoFrame output = core::RawVideoFrame::create_empty(format,{width*x_, height*y_}, true);
	uint8_t* out = PLANE_RAW_DATA(output,0);
	size_t sub_line_width=bpp*width/8;
	size_t line_width = sub_line_width*x_;
	size_t idx = 0;
	for (size_t idx_y=0;idx_y<y_;++idx_y) {
		for (size_t idx_x=0;idx_x<x_;++idx_x) {
			const uint8_t* raw_src = PLANE_RAW_DATA(frames[idx],0);
			for (size_t line=0;line<height;++line) {
				std::copy(raw_src+line*sub_line_width,
						raw_src+(line+1)*sub_line_width,
						out+(idx_y*height+line)*line_width+idx_x*sub_line_width);
			}
			idx++;
		}
	}
	return {output};
//	push_raw_video_frame(0,output);
//	for (size_t i=0;i<frames_no;++i) {
//		frames[i].reset();
//	}
//	return true;
}
bool Combine::set_param(const core::Parameter& param)
{
	if (param.get_name() == "x") {
		x_ = param.get<size_t>();
	} else if (param.get_name() == "y") {
		y_ = param.get<size_t>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
