/*!
 * @file 		EventInfo.h
 * @author 		<Your name>
 * @date 		11.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef EVENTINFO_H_
#define EVENTINFO_H_

#include "yuri/core/BasicIOThread.h"
#include "yuri/event/BasicEventConsumer.h"
namespace yuri {
namespace event_info {

class EventInfo: public core::BasicIOThread, public event::BasicEventConsumer
{
public:
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters 	configure();
	virtual 					~EventInfo();
private:
								EventInfo(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters);
	//virtual bool 				step();
	virtual void				run();
	virtual bool 				set_param(const core::Parameter& param);
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event);
};

} /* namespace event_info */
} /* namespace yuri */
#endif /* EVENTINFO_H_ */
