/*!
 * @file 		EventFrame.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		09.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "EventFrame.h"
#include "misc_frame_types.h"

namespace yuri {
namespace core {

EventFrame::EventFrame(std::string name, event::pBasicEvent event)
:Frame(misc_frames::event_frame),
 name_(std::move(name)),event_(std::move(event))
{

}

EventFrame::~EventFrame() noexcept
{

}

void EventFrame::set_name(std::string name)
{
	name_ = std::move(name);
}

void EventFrame::set_event(event::pBasicEvent event)
{
	event_ = std::move(event);
}

void EventFrame::copy_parameters(Frame& other) const
{
	try {
		auto& frame = dynamic_cast<EventFrame&>(other);
		frame.set_name(get_name());
		frame.set_event(get_event());
	}
	catch (std::bad_cast&)
	{}
	Frame::copy_parameters(other);
}
pFrame EventFrame::do_get_copy() const
{
	return std::make_shared<EventFrame>(name_, event_);
}

size_t EventFrame::do_get_size() const noexcept
{
	return 8;
}

}
}



