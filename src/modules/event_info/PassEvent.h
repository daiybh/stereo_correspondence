/*!
 * @file 		PassEvent.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		11.07.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef PASSEVENT_H_
#define PASSEVENT_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/event/BasicEventProducer.h"
namespace yuri {
namespace pass_events{

class PassEvent: public core::IOThread, public event::BasicEventConsumer, public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters 	configure();
								PassEvent(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual 					~PassEvent() noexcept;
private:
	virtual void				run();
	virtual bool 				set_param(const core::Parameter& param);
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event);
};

} /* namespace event_info */
} /* namespace yuri */
#endif /* PASSEVENT_H_ */
