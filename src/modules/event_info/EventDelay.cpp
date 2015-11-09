/*!
 * @file 		EventDelay.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		09.11.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "EventDelay.h"
#include "yuri/core/Module.h"
#include "yuri/core/utils/assign_events.h"
#include <cassert>
namespace yuri {
namespace event_delay {

IOTHREAD_GENERATOR(EventDelay)

core::Parameters EventDelay::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Timer");
	p["delay"]["Delay in seconds"]=1.0;
	return p;
}


EventDelay::EventDelay(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,0,0,std::string("event_delay")),
event::BasicEventConsumer(log),event::BasicEventProducer(log),
delay_(1_s)
{
	set_latency(5_ms);
	IOTHREAD_INIT(parameters);

}

EventDelay::~EventDelay() noexcept
{
}

void EventDelay::run()
{
	while (still_running()) {
		process_events();
		auto t = timestamp_t{} - delay_;
		while (!events_.empty()) {
			auto& e = events_.front();
			if (std::get<0>(e) > t) {
				break;
			}
			emit_event(std::move(std::get<1>(e)), std::move(std::get<2>(e)));
			events_.pop_front();
		}
		wait_for_events(get_latency());
	}
}

bool EventDelay::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(delay_, "delay", [](const core::Parameter& p){ return 1_s * p.get<double>();}))
		return true;
	return core::IOThread::set_param(param);
}


bool EventDelay::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	events_.push_back(std::make_tuple(timestamp_t{}, event_name, event));
	return true;
}
} /* namespace event_info */
} /* namespace yuri */

