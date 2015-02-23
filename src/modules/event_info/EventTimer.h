/*!
 * @file 		EvenTimer.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		01.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef EVENT_TIMER_H_
#define EVENT_TIMER_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/event/BasicEventProducer.h"
#include "yuri/core/utils/Timer.h"

namespace yuri {
namespace event_timer {

class EventTimer: public core::IOThread, public event::BasicEventConsumer, public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters 	configure();
								EventTimer(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual 					~EventTimer() noexcept;
private:
	virtual void				run();
	virtual bool 				set_param(const core::Parameter& param);
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event);

	void 						emit_events(duration_t);
	duration_t interval_;
	duration_t last_point_;
	index_t index_;
	Timer timer_;
};

} /* namespace event_timer */
} /* namespace yuri */
#endif /* EVENT_TIMER_H_ */

