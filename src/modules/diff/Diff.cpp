/*!
 * @file 		Diff.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Diff.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace diff {

REGISTER("diff",Diff)

IO_THREAD_GENERATOR(Diff)

// So we can write log[info] instead of log[log::info]
using namespace yuri::log;

core::pParameters Diff::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("Image difference.");
	p->set_max_pipes(2,1);
	return p;
}


Diff::Diff(log::Log &log_,core::pwThreadBase parent,core::Parameters &parameters):
core::BasicIOThread(log_,parent,2,1,std::string("diff"))
{
	IO_THREAD_INIT("Diff")
}

Diff::~Diff()
{
}

bool Diff::step()
{
	if (!in[0] || !in[1]) return true;
	if (!frame1) frame1 = in[0]->pop_frame();
	if (!frame2) frame2 = in[1]->pop_frame();
	if (!frame1 || !frame2) return true;
	if (frame1->get_format() != frame2->get_format()) {
		log[warning] << "Frame types have to match!\n";
		frame1.reset();frame2.reset();
		return true;
	}
	if ((frame1->get_width() != frame2->get_width()) || (frame1->get_height() != frame2->get_height())) {
		log[warning] << "Frame sizes have to match\n";
		frame1.reset();frame2.reset();
		return true;
	}
	if (frame1->get_format() != YURI_FMT_RGB24) {
		log[warning] << "frame is not rgb24\n";
		frame1.reset();frame2.reset();
		return true;
	}
	const size_t width = frame1->get_width();
	const size_t height = frame1->get_height();
	core::pBasicFrame output = allocate_empty_frame(YURI_FMT_RGB24,width,height);
	ubyte_t *first = PLANE_RAW_DATA(frame1,0);
	ubyte_t *second= PLANE_RAW_DATA(frame2,0);
	ubyte_t *out_ptr= PLANE_RAW_DATA(output,0);
	for (size_t i=0;i<width*height*3;++i) {
		*out_ptr++=std::abs(static_cast<int>(*first++) - static_cast<int>(*second++));
	}
	push_raw_video_frame(0, output);

	frame1.reset();frame2.reset();
	return true;
}
//bool Diff::set_param(config::Parameter& param)
//{
//	return core::BasicIOThread::set_param(param);
//}

} /* namespace dummy_module */
} /* namespace yuri */
