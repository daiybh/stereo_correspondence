/*!
 * @file 		Diff.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Diff.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"

namespace yuri {
namespace diff {


IOTHREAD_GENERATOR(Diff)

MODULE_REGISTRATION_BEGIN("diff")
		REGISTER_IOTHREAD("diff",Diff)
MODULE_REGISTRATION_END()

// So we can write log[info] instead of log[log::info]
using namespace yuri::log;

core::Parameters Diff::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Image difference.");
//	p->set_max_pipes(2,1);
	return p;
}


Diff::Diff(const log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,2,1,std::string("diff"))
{
	IOTHREAD_INIT(parameters)
}

Diff::~Diff() noexcept
{
}

bool Diff::step()
{
	//if (!in[0] || !in[1]) return true;
	if (!frame1) frame1 = dynamic_pointer_cast<core::RawVideoFrame>(pop_frame(0));
	if (!frame2) frame2 = dynamic_pointer_cast<core::RawVideoFrame>(pop_frame(1));
	if (!frame1 || !frame2) return true;
	if (frame1->get_format() != frame2->get_format()) {
		log[warning] << "Frame types have to match!\n";
		frame1.reset();frame2.reset();
		return true;
	}
	//if ((frame1->get_width() != frame2->get_width()) || (frame1->get_height() != frame2->get_height())) {
	if (frame1->get_resolution() != frame2->get_resolution()) {
		log[warning] << "Frame sizes have to match\n";
		frame1.reset();frame2.reset();
		return true;
	}
	if (frame1->get_format() != core::raw_format::rgb24) {
		log[warning] << "frame is not rgb24\n";
		frame1.reset();frame2.reset();
		return true;
	}
//	const size_t width = frame1->get_width();
//	const size_t height = frame1->get_height();
//	core::pBasicFrame output = allocate_empty_frame(YURI_FMT_RGB24,width,height);
	core::pRawVideoFrame output = core::RawVideoFrame::create_empty(core::raw_format::rgb24, frame1->get_resolution());
	auto first = PLANE_DATA(frame1,0).begin();
	auto first_end = PLANE_DATA(frame1,0).end();
	auto second= PLANE_DATA(frame2,0).begin();
	auto out_ptr= PLANE_DATA(output,0).begin();
	std::transform(first, first_end, second, out_ptr,
			[](const uint8_t&a, const uint8_t&b){
		return static_cast<uint8_t>(std::abs(static_cast<int>(a) - static_cast<int>(b)));
	});
//	for (size_t i=0;i<width*height*3;++i) {
//		*out_ptr++=std::abs(static_cast<int>(*first++) - static_cast<int>(*second++));
//	}
	push_frame(0, output);

	frame1.reset();frame2.reset();
	return true;
}


} /* namespace dummy_module */
} /* namespace yuri */
