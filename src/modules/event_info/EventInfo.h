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

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
namespace yuri {
namespace event_info {

class EventInfo: public core::IOThread, public event::BasicEventConsumer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters 	configure();
								EventInfo(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual 					~EventInfo() noexcept;
private:

	//virtual bool 				step();
	virtual void				run();
	virtual bool 				set_param(const core::Parameter& param);
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event);
};

} /* namespace event_info */
} /* namespace yuri */
#endif /* EVENTINFO_H_ */
