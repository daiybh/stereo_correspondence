/*!
 * @file 		Fade.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		13.07.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Fade.h"
#include "yuri/core/Module.h"
#include "yuri/event/BasicEventConversions.h"
#include "yuri/core/utils/assign_events.h"
namespace yuri {
namespace fade {

IOTHREAD_GENERATOR(Fade)

MODULE_REGISTRATION_BEGIN("fade")
		REGISTER_IOTHREAD("fade",Fade)
MODULE_REGISTRATION_END()

core::Parameters Fade::configure()
{
	core::Parameters p = core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>::configure();
	p.set_description("Fade");
	return p;
}


Fade::Fade(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>(log_, parent, 1, std::string("fade")),
event::BasicEventConsumer(log),transition_(0.0)
{
	IOTHREAD_INIT(parameters)
}

Fade::~Fade() noexcept
{
}
//template<typename T>
//T crop(const T& val, const T& min_, const T& max_)
//{
//
//}
//std::vector<core::pFrame> Fade::do_single_step(const std::vector<core::pFrame>& frames)
std::vector<core::pFrame> Fade::do_special_step(const std::tuple<core::pRawVideoFrame, core::pRawVideoFrame>& frames)
{
	process_events();
//	timestamp_t start_time;
//	if (frames.size() != 2) return {};
	if (std::get<0>(frames)->get_format() != std::get<1>(frames)->get_format()) return {};
	if (PLANE_SIZE(std::get<0>(frames),0) != PLANE_SIZE(std::get<1>(frames),0)) return {};
	resolution_t res = {std::get<0>(frames)->get_width(), std::get<0>(frames)->get_height()};
	core::pRawVideoFrame outframe = core::RawVideoFrame::create_empty(std::get<0>(frames)->get_format(), res);
	auto it0 = PLANE_DATA(std::get<0>(frames),0).begin();
	auto it1 = PLANE_DATA(std::get<1>(frames),0).begin();
	auto it_out = PLANE_DATA(outframe,0).begin();
	auto it_last = PLANE_DATA(outframe,0).end();

//	const double t1 = 1.0 - transition_;
//	while (it_out != it_last) {
//		*it_out++ = static_cast<uint8_t>(t1 * static_cast<double>(*it0++) +
//										transition_ * static_cast<double>(*it1++));
//	}

	const uint_fast16_t trans = static_cast<uint_fast16_t>(transition_*256);
	const uint_fast16_t trans1 = 256- trans;
	while (it_out != it_last) {
		*it_out++ = static_cast<uint8_t>((trans1 * *it0++ +
										trans * *it1++)/256);
	}

//	timestamp_t end_time;
//	log[log::info] << "Fade took: " << (end_time - start_time);
	return {outframe};
}
bool Fade::set_param(const core::Parameter& param)
{
	return core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>::set_param(param);
}
bool Fade::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (assign_events(event_name, event)
			.ranged(transition_, 0.0, 1.0, "state"))
		return true;
	return false;
}


} /* namespace fade */
} /* namespace yuri */
