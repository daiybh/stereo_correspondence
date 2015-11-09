/*!
 * @file 		EvenTimer.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		01.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef EVENT_DELAY_H_
#define EVENT_DELAY_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/event/BasicEventProducer.h"
//#include "yuri/core/utils/Timer.h"

namespace yuri {
namespace event_delay {

class EventDelay: public core::IOThread, public event::BasicEventConsumer, public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters 	configure();
								EventDelay(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual 					~EventDelay() noexcept;
private:
	virtual void				run();
	virtual bool 				set_param(const core::Parameter& param);
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event);

	duration_t delay_;
	using event_record_t = std::tuple<timestamp_t, std::string, event::pBasicEvent>;
	std::deque<event_record_t> events_;
};

} /* namespace event_delay */
} /* namespace yuri */
#endif /* EVENT_DELAY_H_ */

