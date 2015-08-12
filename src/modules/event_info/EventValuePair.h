/*!
 * @file 		EventValuePair.h
 * @author 		Jiri Melnikov <melnikov@cesnet.cz>
 * @date 		11.06.2015
 * @copyright	CESNET z.s.p.o.
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef EVENTVALUEPAIR_H_
#define EVENTVALUEPAIR_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/event/BasicEventProducer.h"

namespace yuri {
namespace event_value_pair {

class EventValuePair: public core::IOThread, public event::BasicEventConsumer, public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters 	configure();
								EventValuePair(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual 					~EventValuePair() noexcept;
private:
	virtual void				run();
	virtual bool 				set_param(const core::Parameter& param);
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event);

	std::string 				line_event_name_;
	std::string 				raw_separator_;
	std::string 				value_separator_;

};

} /* namespace event_value_pair */
} /* namespace yuri */
#endif /* EVENTVALUEPAIR_H_ */
