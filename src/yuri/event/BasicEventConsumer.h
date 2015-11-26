/*!
 * @file 		BasicEventConsumer.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		03.07.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef BASICEVENTCONSUMER_H_
#define BASICEVENTCONSUMER_H_
#include "yuri/event/BasicEvent.h"
#include "yuri/log/Log.h"
#include <deque>
#include "yuri/core/utils/time_types.h"

namespace yuri {
namespace event {
typedef std::pair<std::string, const pBasicEvent>
								event_record_t;
class BasicEventConsumer;

using pBasicEventConsumer  = std::shared_ptr<BasicEventConsumer>;	
using pwBasicEventConsumer = std::weak_ptr<BasicEventConsumer>;

class BasicEventConsumer {
public:
	EXPORT 						BasicEventConsumer(log::Log&);
	EXPORT virtual				~BasicEventConsumer();
	EXPORT bool 				receive_event(const std::string& event_name, const pBasicEvent& event);
protected:
	EXPORT event_record_t		get_pending_event();
	EXPORT virtual void 		receive_event_hook() noexcept {};

	EXPORT bool 				process_events(ssize_t max_count = -1);
	EXPORT size_t				pending_events() const;
	EXPORT bool					wait_for_events(duration_t timeout);
private:
	EXPORT bool 				do_receive_event(const std::string& event_name, const pBasicEvent& event);
	EXPORT bool 				do_process_events(ssize_t max_count);
	virtual bool 				do_process_event(const std::string& event_name, const pBasicEvent& event) = 0;
	std::deque<event_record_t> 	incomming_events_;
	mutable mutex				incomming_mutex_;
	size_t						incomming_max_size_ = 1024;
	log::Log					log_c_;
	std::condition_variable		incomming_notification_;
	size_t						lost_events_;
};

}
}


#endif /* BASICEVENTCONSUMER_H_ */
