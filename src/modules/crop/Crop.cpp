/*!
 * @file 		Crop.cpp
 * @author 		Zdenek Travnicek
 * @date 		17.11.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Crop.h"
#include "yuri/core/Module.h"

namespace yuri {

namespace io {

REGISTER("crop",Crop)

IO_THREAD_GENERATOR(Crop)

core::pParameters Crop::configure()
{
	core::pParameters p (new core::Parameters());
	(*p)["x"]["offset x"]=0;
	(*p)["y"]["offset y"]=0;
	(*p)["width"]["width"]=-1;
	(*p)["height"]["height"]=-1;
	p->set_max_pipes(1,1);
//	p->add_input_format(YURI_FMT_RGB);
//	p->add_output_format(YURI_FMT_RGB);
	return p;
}


Crop::Crop(log::Log &_log, core::pwThreadBase parent,core::Parameters &parameters) IO_THREAD_CONSTRUCTOR:
	core::BasicIOThread(_log,parent,1,1)
{
	IO_THREAD_INIT("Crop")
}

Crop::~Crop() {

}

bool Crop::step()
{
	core::pBasicFrame frame;
	if (!in[0] || !(frame = in[0]->pop_frame()))
		return true;

	const size_t in_width = frame->get_width();
	const size_t in_height = frame->get_height();
	yuri::ssize_t x = dest_x, y = dest_y, w = dest_w, h = dest_h;
	if (x >= static_cast<yuri::ssize_t>(in_width) || y >= static_cast<yuri::ssize_t>(in_height))
		return true;
	if (w < 0) {
		w = in_width - x;
	} else 	if (x + w > static_cast<yuri::ssize_t>(in_width)) {
		w = in_width - x;
	}
	if (h < 0) {
		h = static_cast<yuri::ssize_t>(in_height) - y;
	} else if (y + h > static_cast<yuri::ssize_t>(in_height)) {
		h = in_height - y;
	}
	log[log::verbose_debug] << "X: " << x << ", Y: " << y << ", W: " << w<< ", H: " << h;
	if (!x && !y && w==static_cast<yuri::ssize_t>(in_width)
			&& h==static_cast<yuri::ssize_t>(in_height)) {
		log[log::verbose_debug] << "Passing thru";
		push_raw_video_frame(0,frame);
		return true;
	}

	core::pBasicFrame frame_out = allocate_empty_frame(frame->get_format(),w, h, true);
	const FormatInfo_t info = core::BasicPipe::get_format_info(frame->get_format());
	assert(info);
	if (info->planes!=1) {
		log[log::warning] << "Received frame has more that a single plane. \n";
		return true;
	}
	if (info->bpp&0x07) {
		log[log::warning] << "Currently only formats with bit depth divisible by 8 are supported.";
		return true;
	}
	yuri::size_t Bpp = info->bpp >> 3;
	log[log::verbose_debug] << "size: " << w <<"x"<<h<<"+"<<x<<"+"<<y<<" at "<<Bpp<<"Bpp"<<std::endl;
	yuri::ubyte_t  *out_ptr=PLANE_RAW_DATA(frame_out,0);
	const size_t in_line_width = in_width * Bpp;
	const size_t out_line_width = w * Bpp;
	log[log::verbose_debug] << "in_line_width: " << in_line_width << ", out_line_width: " << out_line_width;
	const yuri::ubyte_t *ptr = PLANE_RAW_DATA(frame,0)+(y * in_line_width) + x*Bpp;
	for (int i = 0; i < h; ++i) {
		std::copy(ptr, ptr + out_line_width, out_ptr);
		out_ptr+=out_line_width;
		ptr+=in_line_width;
	}
	push_video_frame(0,frame_out,frame->get_format(),w, h);
	return true;
}

bool Crop::set_param(const core::Parameter &parameter)
{
	if (parameter.name== "x") {
		dest_x=parameter.get<yuri::ssize_t>();
	} else if (parameter.name== "y") {
		dest_y=parameter.get<yuri::ssize_t>();
	} else if (parameter.name== "width") {
		dest_w=parameter.get<yuri::ssize_t>();
	} else if (parameter.name== "height") {
		dest_h=parameter.get<yuri::ssize_t>();
	} else  return BasicIOThread::set_param(parameter);
	return true;
}
}
}
