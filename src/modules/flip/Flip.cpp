/*!
 * @file 		Flip.cpp
 * @author 		Zdenek Travnicek
 * @date 		16.3.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2012 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Flip.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/raw_frame_types.h"
#include <cassert>
namespace yuri {

namespace io {

IOTHREAD_GENERATOR(Flip)

MODULE_REGISTRATION_BEGIN("flip")
		REGISTER_IOTHREAD("flip",Flip)
MODULE_REGISTRATION_END()

core::Parameters Flip::configure()
{
	core::Parameters p = IOThread::configure();
	p["flip_x"]["flip x (around y axis)"]=true;
	p["flip_y"]["flip y (around X axis)"]=false;
	//p->set_max_pipes(1,1);
	//p->add_input_format(YURI_FMT_RGB);
	//p->add_output_format(YURI_FMT_RGB);
	return p;
}


Flip::Flip(log::Log &_log, core::pwThreadBase parent, const core::Parameters &parameters)
:core::SpecializedIOFilter<core::RawVideoFrame>(_log,parent,"flip"),
 flip_x_(true),flip_y_(false)
 {
	IOTHREAD_INIT(parameters)

}

Flip::~Flip() noexcept {

}

core::pFrame Flip::do_special_single_step(const core::pRawVideoFrame& frame)
//bool Flip::step()
{
//	core::pBasicFrame  frame;
//	if (!in[0] || !(frame = in[0]->pop_frame()))
//		return true;

	yuri::size_t	w = frame->get_width();
	yuri::size_t	h = frame->get_height();


	auto frame_copy = frame->get_copy();
	core::pRawVideoFrame  frame_out = dynamic_pointer_cast<core::RawVideoFrame>(frame_copy);
	assert(frame_out);
//	FormatInfo_t info = core::BasicPipe::get_format_info(frame->get_format());
//	assert(info && info->planes==1);
	const auto& fi = core::raw_format::get_format_info(frame->get_format());
	assert(fi.planes.size() == 1);
	yuri::size_t Bpp = fi.planes[0].bit_depth.first/fi.planes[0].bit_depth.second/8;
	yuri::size_t line_length = w*Bpp;
	bool swap_l = false;
	if (frame->get_format()==core::raw_format::yuyv422) {
		Bpp*=2;
		swap_l=true;
	}
	if (flip_y_) log[log::warning] << "Flip_y not supported!!";
	if (flip_x_) {
		uint8_t *base_ptr =  PLANE_RAW_DATA(frame_out,0);
//		yuri::size_t cnt = 0;
		for (yuri::size_t line =0;line<h;++line) {
			uint8_t  *in_ptr = base_ptr+line*line_length;
			uint8_t  *out_ptr = base_ptr+(line+1)*line_length-Bpp;
//			yuri::ubyte_t t;
			while (out_ptr>in_ptr) {
				for (uint8_t  b=0;b<Bpp;++b) {
					std::swap(*in_ptr++,*out_ptr++);
				}
				if (swap_l) {
					std::swap(*(in_ptr-2),*(in_ptr-4));
					std::swap(*(out_ptr-2),*(out_ptr-4));
				}
				out_ptr-=2*Bpp;
//				t = *in_ptr;
//				*in_ptr++ = *out_ptr;
//				*out_ptr--=t;
//				cnt++;
				//std::swap(*in_ptr++,*out_ptr--);
			}
		}
//		log[log::yuri::log::info] << "swapped " << cnt << " values" << std::endl;
	}
	return frame_out;
//	push_raw_video_frame(0,frame_out);
//	return true;
}

bool Flip::set_param(const core::Parameter &parameter)
{
	if (parameter.get_name()== "flip_x") {
		flip_x_=parameter.get<bool>();
//		log[log::info] << "flip_x is: " << flip_x_<<std::endl;
	} else if (parameter.get_name()== "flip_y") {
		flip_y_=parameter.get<bool>();
	} else return core::SpecializedIOFilter<core::RawVideoFrame>::set_param(parameter);
	return true;
}

}

}
