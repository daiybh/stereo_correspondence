/*!
 * @file 		EventTimer.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		01.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "EventTimer.h"
#include "yuri/core/Module.h"
#include "yuri/core/utils/assign_events.h"
#include <cassert>
namespace yuri {
namespace event_timer {

IOTHREAD_GENERATOR(EventTimer)

core::Parameters EventTimer::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Timer");
	p["interval"]["Interval to send notifications (in seconds)"]=1.0;
	return p;
}


EventTimer::EventTimer(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,0,0,std::string("event_timer")),
event::BasicEventConsumer(log),event::BasicEventProducer(log),
interval_(1_s),index_(0)
{
	set_latency(10_ms);
	IOTHREAD_INIT(parameters);
	if (get_latency() > interval_) {
		set_latency(interval_/4);
	}
}

EventTimer::~EventTimer() noexcept
{
}

void EventTimer::run()
{
	timer_.reset();
	while (still_running()) {
		wait_for_events(get_latency());
		const auto dur = timer_.get_duration();
		if ((dur - last_point_) > interval_) {
			emit_events(dur);
			last_point_+=interval_;
			index_++;
		}
		process_events();
	}
}

void EventTimer::emit_events(duration_t dur)
{
	emit_event("timer");
	emit_event("timer_string", lexical_cast<std::string>(dur));
	emit_event("index",index_);
}
bool EventTimer::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(interval_, "interval", [](const core::Parameter& p){ return 1_s * p.get<double>();}))
		return true;
	return core::IOThread::set_param(param);
}


bool EventTimer::do_process_event(const std::string& event_name, const event::pBasicEvent& /* event */)
{
	if (event_name == "reset") {
		timer_.reset();
		last_point_ = {};
		index_ = 0;
		emit_events(0_s);
	}
	return true;
}
} /* namespace event_info */
} /* namespace yuri */

