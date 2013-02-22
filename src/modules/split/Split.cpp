/*!
 * @file 		Split.cpp
 * @author 		Zdenek Travnicek
 * @date 		30.3.2011
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2011 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Split.h"
#include "yuri/core/Module.h"

namespace yuri {

namespace split {


REGISTER("split",Split)

IO_THREAD_GENERATOR(Split)

core::pParameters Split::configure()
{
	core::pParameters p (new core::Parameters());
	/*(*p)["x"]["offset x"]=0;
	(*p)["y"]["offset y"]=0;
	(*p)["width"]["width"]=-1;
	(*p)["height"]["height"]=-1;*/
	p->set_max_pipes(1,2);
	p->add_input_format(YURI_FMT_RGB);
	p->add_output_format(YURI_FMT_RGB);
	return p;
}


Split::Split(log::Log &_log, core::pwThreadBase parent,core::Parameters &par):
			BasicIOThread(_log,parent,1,2)
{
	params.merge(par);
}

Split::~Split() {
}

bool Split::step()
{
	core::pBasicFrame frame;
	if (!in[0] || !(frame = in[0]->pop_frame()))
		return true;

	int left, right;
	left = (frame->get_width() / 2)&~0x1;
	right = frame->get_width() - left;
	yuri::size_t height = frame->get_height();
	core::pBasicFrame frame_out1 = allocate_empty_frame(frame->get_format(),left, height);
	core::pBasicFrame frame_out2 = allocate_empty_frame(frame->get_format(),right, height);
	FormatInfo_t info = core::BasicPipe::get_format_info(frame->get_format());
	yuri::size_t Bpp = info->bpp >> 3;
	//log[verbose_debug] << "size: " << dest_w <<"x"<<dest_h<<"+"<<dest_x<<"+"<<dest_y<<" at "<<Bpp<<"Bpp"<<std::endl;
	yuri::ubyte_t *out_ptr1=PLANE_RAW_DATA(frame_out1,0);
	yuri::ubyte_t *out_ptr2=PLANE_RAW_DATA(frame_out2,0);
	yuri::ubyte_t *ptr = PLANE_RAW_DATA(frame,0);
	for (yuri::size_t i=0;i<height;++i) {
		memcpy(out_ptr1,ptr,Bpp*left);
		ptr+=Bpp*left; out_ptr1+=Bpp*left;
		memcpy(out_ptr2,ptr,Bpp*right);
		ptr+=Bpp*right; out_ptr2+=Bpp*right;
	}
	push_video_frame(0,frame_out1,frame->get_format(),left, height);
	push_video_frame(1,frame_out2,frame->get_format(),right, height);
	return true;
}

}

}
