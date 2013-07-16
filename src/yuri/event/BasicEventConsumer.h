/*!
 * @file 		BasicEventConsumer.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		03.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef BASICEVENTCONSUMER_H_
#define BASICEVENTCONSUMER_H_
#include "yuri/event/BasicEvent.h"
#include "yuri/log/Log.h"
#include <deque>
namespace yuri {
namespace event {
typedef std::pair<std::string, const pBasicEvent>
								event_record_t;
class BasicEventConsumer;
typedef shared_ptr<BasicEventConsumer>
								pBasicEventConsumer;
typedef weak_ptr<BasicEventConsumer>
								pwBasicEventConsumer;

class BasicEventConsumer {
public:
								BasicEventConsumer(log::Log&);
	virtual						~BasicEventConsumer();
	bool 						receive_event(const std::string& event_name, const pBasicEvent& event);
protected:
	event_record_t 				fetch_event();
	bool 						process_events(ssize_t max_count = -1);
	size_t						pending_events() const;
private:
	bool 						do_receive_event(const std::string& event_name, const pBasicEvent& event);
	bool 						do_process_events(ssize_t max_count);
	virtual bool 				do_process_event(const std::string& event_name, const pBasicEvent& event) = 0;
	virtual void				do_report_consumer_error(const std::string&) {}
	std::deque<event_record_t> 	incomming_events_;
	mutable mutex				incomming_mutex_;
	size_t						incomming_max_size_ = 1024;
	log::Log&					log_c_;
};

}
}


#endif /* BASICEVENTCONSUMER_H_ */
