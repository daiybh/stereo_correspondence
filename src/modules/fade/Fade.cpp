/*!
 * @file 		Fade.cpp
 * @author 		<Your name>
 * @date		13.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "Fade.h"
#include "yuri/core/Module.h"
#include "yuri/event/BasicEventConversions.h"
namespace yuri {
namespace fade {

REGISTER("fade",Fade)

IO_THREAD_GENERATOR(Fade)

core::pParameters Fade::configure()
{
	core::pParameters p = core::BasicMultiIOFilter::configure();
	p->set_description("Fade");
	p->set_max_pipes(2,1);
	return p;
}


Fade::Fade(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicMultiIOFilter(log_,parent,2,1,std::string("fade")),
event::BasicEventConsumer(),transition_(0.0)
{
	IO_THREAD_INIT("fade")
}

Fade::~Fade()
{
}
//template<typename T>
//T crop(const T& val, const T& min_, const T& max_)
//{
//
//}
std::vector<core::pBasicFrame> Fade::do_single_step(const std::vector<core::pBasicFrame>& frames)
{
	process_events();
	if (frames.size() != 2) return {};
	if (frames[0]->get_format() != frames[1]->get_format()) return {};
	if (PLANE_SIZE(frames[0],0) != PLANE_SIZE(frames[1],0)) return {};
	core::pBasicFrame outframe = allocate_empty_frame(frames[0]->get_format(), frames[0]->get_width(), frames[0]->get_height());
	auto it0 = PLANE_DATA(frames[0],0).begin();
	auto it1 = PLANE_DATA(frames[1],0).begin();
	auto it_out = PLANE_DATA(outframe,0).begin();
	auto it_last = PLANE_DATA(outframe,0).end();
	const double t1 = 1.0 - transition_;
	while (it_out != it_last) {
		*it_out++ = static_cast<ubyte_t>(t1 * static_cast<double>(*it0++) +
										transition_ * static_cast<double>(*it1++));
	}
	return {outframe};
}
bool Fade::set_param(const core::Parameter& param)
{
	return core::BasicIOThread::set_param(param);
}
bool Fade::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (event_name == "state") {
		transition_ = event::get_value<event::EventDouble>(event);
		transition_ = std::min(std::max(transition_,0.0),1.0);
		log[log::info] << "Set transition to " << transition_;
	}
	return true;
}


} /* namespace fade */
} /* namespace yuri */
