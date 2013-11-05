/*!
 * @file 		BasicEventProducer.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		03.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef BASICEVENTPRODUCER_H_
#define BASICEVENTPRODUCER_H_
#include "BasicEventConsumer.h"
#include <unordered_map>
#include <deque>
namespace yuri {
namespace event {
typedef std::pair<pwBasicEventConsumer, const std::string>
								event_target_t;
class BasicEventProducer;
typedef shared_ptr<BasicEventProducer>
								pBasicEventProducer;

typedef std::pair<std::string, event_type_t>
								event_info_t;

class BasicEventProducer {
public:
								BasicEventProducer(log::Log&);
	virtual						~BasicEventProducer();
	bool 						register_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name);
	bool 						unregister_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name);
	std::vector<event_info_t>	list_events();
protected:
	bool 						emit_event(const std::string& event_name, pBasicEvent event);

	// Overload for emitting a bang event
	bool 						emit_event(const std::string& event_name) {
		return emit_event(event_name, make_shared<EventBang>());
	}
	// Overload for emitting a boolean event
	bool 						emit_event(const std::string& event_name, bool value) {
		return emit_event(event_name, make_shared<EventBool>(value));
	}
	// Overload for emitting a string event
	bool						emit_event(const std::string& event_name, const std::string& value)
	{
		return emit_event(event_name, make_shared<EventString>(value));
	}
	// Overload for emitting a time event
	bool						emit_event(const std::string& event_name, timestamp_t value)
	{
		return emit_event(event_name, make_shared<EventTime>(value));
	}
	// Overload for emitting an integer event
	template<typename T>
	typename std::enable_if<std::is_integral<T>::value, bool>::type
								emit_event(const std::string& event_name, T value)
	{
		return emit_event(event_name, make_shared<EventInt>(value));
	}
	// Overload for emitting a double event
	template<typename T>
	typename std::enable_if<std::is_floating_point<T>::value, bool>::type
								emit_event(const std::string& event_name, T value)
	{
		return emit_event(event_name, make_shared<EventDouble>(value));
	}


	// Overload for emitting an integer event with range
	template<typename T, typename T2, typename T3>
	typename std::enable_if<std::is_integral<T>::value && std::is_integral<T2>::value
			&& std::is_integral<T3>::value, bool>::type
								emit_event(const std::string& event_name, T value, T2 min_value, T3 max_value)
	{
		return emit_event(event_name, make_shared<EventInt>(value, min_value, max_value));
	}
	// Overload for emitting a double event
	template<typename T, typename T2, typename T3>
	typename std::enable_if<std::is_floating_point<T>::value && std::is_floating_point<T2>::value
			&& std::is_floating_point<T3>::value, bool>::type
								emit_event(const std::string& event_name, T value, T min_value, T max_value)
	{
		return emit_event(event_name, make_shared<EventDouble>(value, min_value, max_value));
	}


private:
	bool 						do_register_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name);
	bool 						do_unregister_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name);
	virtual	std::vector<event_info_t>
								do_list_events() { return {}; }
	virtual bool				verify_register_event(const::std::string& event_name);
	std::unordered_multimap<std::string, event_target_t>
								consumers_;
	mutex						consumers_mutex_;
	log::Log&					log_p_;

};


}
}

#endif /* BASICEVENTPRODUCER_H_ */
