/*
 * Crop.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: neneko
 */

#include "Crop.h"


namespace yuri {

namespace io {

REGISTER("crop",Crop)

IO_THREAD_GENERATOR(Crop)

shared_ptr<Parameters> Crop::configure()
{
	shared_ptr<Parameters> p (new Parameters());
	(*p)["x"]["offset x"]=0;
	(*p)["y"]["offset y"]=0;
	(*p)["width"]["width"]=-1;
	(*p)["height"]["height"]=-1;
	p->set_max_pipes(1,1);
	p->add_input_format(YURI_FMT_RGB);
	p->add_output_format(YURI_FMT_RGB);
	return p;
}


Crop::Crop(Log &_log, pThreadBase parent, Parameters &parameters) IO_THREAD_CONSTRUCTOR:
	BasicIOThread(_log,parent,1,1)
{
	IO_THREAD_INIT("Crop")
}

Crop::~Crop() {

}

bool Crop::step()
{
	shared_ptr<BasicFrame> frame;
	if (!in[0] || !(frame = in[0]->pop_frame()))
		return true;

	yuri::ssize_t x = dest_x, y = dest_y, w = dest_w, h = dest_h;
	if (x >= static_cast<yuri::ssize_t>(frame->get_width()) || y >= static_cast<yuri::ssize_t>(frame->get_height()))
		return true;
	if (w < 0) {
		w = frame->get_width() - x;
	} else 	if (x + w > static_cast<yuri::ssize_t>(frame->get_width())) {
		w = frame->get_width() - x;
	}
	if (h < 0) {
		h = static_cast<yuri::ssize_t>(frame->get_height()) - y;
	} else if (y + h > static_cast<yuri::ssize_t>(frame->get_height())) {
		h = frame->get_width() - y;
	}
	log[verbose_debug] << "X: " << x << ", Y: " << y << ", W: " << w<< ", H: " << h <<endl;
	if (!x && !y && w==static_cast<yuri::ssize_t>(frame->get_width())
			&& h==static_cast<yuri::ssize_t>(frame->get_height())) {
		log[verbose_debug] << "Passing thru" << endl;
		push_raw_video_frame(0,frame);
		return true;
	}

	shared_ptr<BasicFrame> frame_out = allocate_empty_frame(frame->get_format(),w, h);
	FormatInfo_t info = BasicPipe::get_format_info(frame->get_format());
	assert(info && info->planes==1);
	yuri::size_t Bpp = info->bpp >> 3;
	log[verbose_debug] << "size: " << w <<"x"<<h<<"+"<<x<<"+"<<y<<" at "<<Bpp<<"Bpp"<<endl;
	yuri::ubyte_t  *out_ptr=(*frame_out)[0].data.get();
	for (int i=y;i<(h+y);++i) {
		yuri::ubyte_t *ptr = (*frame)[0].data.get()+(i*frame->get_width()+x)*Bpp;
		for (int j=0;j<(int)(w*Bpp);++j) *out_ptr++ = *ptr++;
	}
	push_video_frame(0,frame_out,frame->get_format(),w, h);
	return true;
}

bool Crop::set_param(Parameter &parameter)
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
