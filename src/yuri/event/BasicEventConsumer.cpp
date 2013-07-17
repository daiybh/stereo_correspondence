/*!
 * @file 		BasicEventConsumer.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		04.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
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
	lock _(incomming_mutex_);
	return do_receive_event(event_name, event);
}

bool BasicEventConsumer::do_receive_event(const std::string& event_name, const pBasicEvent& event)
{
	event_record_t rec{event_name, event};
	incomming_events_.push_back(rec);
	while (incomming_events_.size() > incomming_max_size_) {
		incomming_events_.pop_front();
	}
	return true;
}
bool BasicEventConsumer::process_events(ssize_t max_count)
{
	lock _(incomming_mutex_);
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
			log_c_[log::debug] << "Error while processing incomming event '" << rec.first << e.what();
		}
	}
	return true;
}
size_t	BasicEventConsumer::pending_events() const
{
	lock _(incomming_mutex_);
	return incomming_events_.size();
}


}
}


