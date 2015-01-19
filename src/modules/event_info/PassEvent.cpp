/*!
 * @file 		PassEvent.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		11.07.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "PassEvent.h"
#include "yuri/core/Module.h"
#include <cassert>
namespace yuri {
namespace pass_events {

IOTHREAD_GENERATOR(PassEvent)

core::Parameters PassEvent::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("PassEvent");
//	p->set_max_pipes(1,1);
	return p;
}


PassEvent::PassEvent(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,0,0,std::string("event_info")),
event::BasicEventConsumer(log),event::BasicEventProducer(log)
{
	IOTHREAD_INIT(parameters);
	set_latency(50_ms);
}

PassEvent::~PassEvent() noexcept
{
}

void PassEvent::run()
{
	while (still_running()) {
		wait_for_events(get_latency());
		process_events();
	}
}
bool PassEvent::set_param(const core::Parameter& param)
{
	return core::IOThread::set_param(param);
}


bool PassEvent::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	emit_event(event_name, event);
	return true;
}
} /* namespace event_info */
} /* namespace yuri */
