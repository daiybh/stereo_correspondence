/*!
 * @file 		BasicEventConsumer.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		04.07.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "BasicEventConsumer.h"
namespace yuri {
namespace event {

BasicEventConsumer::BasicEventConsumer(log::Log& log):log_c_(log)
{

}

BasicEventConsumer::~BasicEventConsumer()
{

}


bool BasicEventConsumer::receive_event(const std::string& event_name, const pBasicEvent& event)
{
	lock_t _(incomming_mutex_);
	return do_receive_event(event_name, event);
}

bool BasicEventConsumer::do_receive_event(const std::string& event_name, const pBasicEvent& event)
{
	event_record_t rec{event_name, event};
	incomming_events_.push_back(rec);
	while (incomming_events_.size() > incomming_max_size_) {
		incomming_events_.pop_front();
	}
	incomming_notification_.notify_all();
	return true;
}
bool BasicEventConsumer::process_events(ssize_t max_count)
{
	lock_t _(incomming_mutex_);
	return do_process_events(max_count);
}

bool BasicEventConsumer::do_process_events(ssize_t max_count)
{
	while (incomming_events_.size() && max_count!=0) {
		auto rec = incomming_events_.front();
		incomming_events_.pop_front();

		try {
			//if (!do_process_event(rec.first, rec.second)) return false;
			do_process_event(rec.first, rec.second);
		}
		catch (std::runtime_error& e) {
			log_c_[log::debug] << "Error while processing incomming event '" << rec.first <<"': "<< e.what();
		}
	}
	return true;
}
size_t	BasicEventConsumer::pending_events() const
{
	lock_t _(incomming_mutex_);
	return incomming_events_.size();
}
bool BasicEventConsumer::wait_for_events(duration_t timeout)
{
	lock_t l(incomming_mutex_);
	if (incomming_events_.size()) return true;
	incomming_notification_.wait_for(l, std::chrono::microseconds(timeout));
	return incomming_events_.size() != 0;
}


}
}


