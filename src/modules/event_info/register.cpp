/*!
 * @file 		register.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		15.4.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */
#include "yuri/core/thread/IOThreadGenerator.h"
#include "EventInfo.h"
#include "PassEvent.h"
#include "EventTimer.h"

namespace yuri {

MODULE_REGISTRATION_BEGIN("event_misc")
		REGISTER_IOTHREAD("event_info", event_info::EventInfo)
		REGISTER_IOTHREAD("pass_event", pass_events::PassEvent)
		REGISTER_IOTHREAD("event_timer", event_timer::EventTimer)
MODULE_REGISTRATION_END()

}
