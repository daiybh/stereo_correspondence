/*!
 * @file 		EventToFrame.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		12.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "EventToFrame.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/EventFrame.h"

namespace yuri {
namespace event_to_frame {


IOTHREAD_GENERATOR(EventToFrame)

core::Parameters EventToFrame::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("EventToFrame");
	return p;
}


EventToFrame::EventToFrame(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("event_to_frame")),
event::BasicEventConsumer(log), event::BasicEventProducer(log)
{
	IOTHREAD_INIT(parameters)
}

EventToFrame::~EventToFrame() noexcept
{
}

void EventToFrame::run()
{
	while (still_running()) {
		wait_for(get_latency());
		process_events();
		while (auto frame = pop_frame(0)) {
			if (auto eframe = std::dynamic_pointer_cast<core::EventFrame>(frame)) {
				emit_event(eframe->get_name(), eframe->get_event());
			} else {
				push_frame(0, frame);
			}
		}

	}
}

bool EventToFrame::set_param(const core::Parameter& param)
{
	return core::IOThread::set_param(param);
}

bool EventToFrame::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	auto frame = std::make_shared<core::EventFrame>(event_name, event);
	push_frame(0, frame);
	return true;
}

void EventToFrame::receive_event_hook() noexcept
{
	notify();
}

} /* namespace event_to_frame */
} /* namespace yuri */
