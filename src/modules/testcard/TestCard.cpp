/*!
 * @file 		TestCard.cpp
 * @author 		<Your name>
 * @date		25.09.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "TestCard.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/RawVideoFrame.h"
namespace yuri {
namespace testcard {


IOTHREAD_GENERATOR(TestCard)

MODULE_REGISTRATION_BEGIN("testcard")
		REGISTER_IOTHREAD("testcard",TestCard)
MODULE_REGISTRATION_END()

namespace {
	std::vector<uint32_t> pattern_colors =
	{0xFF0000, 0x00FF00, 0x0000FF,
	 0xFF00FF, 0xFFFF00, 0x00FFFF};
}


core::Parameters TestCard::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("TestCard");
	p["resolution"]["Test pattern resolution"]=resolution_t{800,600};
	p["fps"]["Framerate of the test pattern"]=25.0;
	p["format"]["Format of the test pattern"]="RGB";
	return p;
}


TestCard::TestCard(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("testcard")),resolution_{800, 600},
fps_(25.0),format_(core::raw_format::yuyv422)
{
	IOTHREAD_INIT(parameters)
}

TestCard::~TestCard() noexcept
{
}

void TestCard::run()
{
	while(still_running()) {
		core::pRawVideoFrame frame = core::RawVideoFrame::create_empty(core::raw_format::rgba32, resolution_, true);
		const size_t cnum = pattern_colors.size();
		auto it = PLANE_DATA(frame,0).begin();
		for (dimension_t line = 0; line < resolution_.height; ++line) {
			//dimension_t col = 0;
			for (dimension_t color = 0; color < cnum; ++color) {
				uint32_t c = pattern_colors[color];
				for (dimension_t col = color * resolution_.width / cnum;
						col < (color+1) * resolution_.width / cnum; ++ col) {
						*it++ = (c&0xFF0000) >> 16;
						*it++ = (c&0x00FF00) >>  8;
						*it++ = (c&0x0000FF) >>  0;
						*it++ = 0xFF;
				}
			}
		}
		push_frame(0, frame);
	}
}
bool TestCard::set_param(const core::Parameter& param)
{
	if (iequals(param.get_name(), "resolution")) {
		resolution_ = param.get<resolution_t>();
	} else if (iequals(param.get_name(), "fps")) {
		fps_ = param.get<double>();
	} else if (iequals(param.get_name(), "format")) {
		format_ = core::raw_format::parse_format(param.get<std::string>());
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace testcard */
} /* namespace yuri */
