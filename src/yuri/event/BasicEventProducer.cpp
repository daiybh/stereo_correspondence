/*!
 * @file 		BasicEventProducer.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		04.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "BasicEventProducer.h"

namespace yuri {
namespace event {
BasicEventProducer::~BasicEventProducer()
{

}
namespace {
template<class Cmp = std::owner_less<pwBasicEventConsumer>>
constexpr bool targets_equal(const event_target_t& lhs, const event_target_t& rhs, Cmp cmp = Cmp()) {
	return (lhs.second == rhs.second) &&
			!cmp(lhs.first, rhs.first) &&
			!cmp(rhs.first, lhs.first);
}
}
bool BasicEventProducer::register_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name)
{
	yuri::lock _(consumers_mutex_);
	return do_register_listener(event_name, consumer, target_name);
}

bool BasicEventProducer::do_register_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name)
{
	event_target_t target{consumer, target_name};
	// Try to verify whether this target was already registered
	if (consumers_.count(event_name) > 0) {
		auto range = consumers_.equal_range(event_name);
		for (auto it = range.first; it != range.second; ++it) {
			if (targets_equal(it->second, target)) return false;
		}

	}
	consumers_.insert({event_name, target});
	return true;
}
bool BasicEventProducer::unregister_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name)
{
	yuri::lock _(consumers_mutex_);
	return do_unregister_listener(event_name, consumer, target_name);
}
bool BasicEventProducer::do_unregister_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name)
{
	const event_target_t target{consumer, target_name};
	size_t erased = 0;
	if (consumers_.count(event_name) == 0) return false;
	auto range = consumers_.equal_range(event_name);
	for (auto it = range.first; it != range.second; ++it) {
		if (targets_equal(it->second, target)) {
			consumers_.erase(it);
			erased++;
		}
	}
	return erased > 0;
}
bool BasicEventProducer::emit_event(const std::string& event_name, pBasicEvent event)
{
	if (!event) return false;
	yuri::lock _(consumers_mutex_);
	if (consumers_.count("*")) {
		auto range = consumers_.equal_range("*");
		for (auto it = range.first; it != range.second; ++it) {
			auto& target = it->second;
			auto consumer = target.first.lock();
			if (!consumer) continue;
			if (target.second == "*") {
				consumer->receive_event(event_name, event);
			} else {
				consumer->receive_event(target.second, event);
			}
		}
	}
	if (!consumers_.count(event_name)) return true;
	auto range = consumers_.equal_range(event_name);
	for (auto it = range.first; it != range.second; ++it) {
		auto& target = it->second;
		auto consumer = target.first.lock();
		if (!consumer) {
			// Basic clean up. The consumer ptr has already expired, so it can be removed from consumers_
			consumers_.erase(it);
		} else {
			consumer->receive_event(target.second, event);
		}
	}
	return true;
}
std::vector<event_info_t> BasicEventProducer::list_events()
{
	return do_list_events();
}
bool BasicEventProducer::verify_register_event(const::std::string& event_name)
{
	return event_name.size();
}

}
}
