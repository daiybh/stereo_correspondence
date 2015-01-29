/*!
 * @file 		assign_events.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		29.01.2015
 * @copyright	Institute of Intermedia, 2015
 * 				Distributed BSD License
 *
 */


#ifndef ASSIGN_EVENTS_H_
#define ASSIGN_EVENTS_H_
#include "yuri/event/EventHelpers.h"

namespace yuri {
struct assign_events {
	assign_events(const std::string& event_name, const event::pBasicEvent& event):
		event_name(event_name),event(event), assigned(false) {}

	template<class T>
	assign_events& operator()(T&) {
		return *this;
	}
	template<class T, class Str, class... Args>
	assign_events& operator()(T& target, Str&& name, Args&&... names)
	{
		if (event_name == name) {
			target = event::lex_cast_value<T>(event);
			assigned = true;
		} else {
			return (*this)(target, names...);
		}
		return *this;
	}


	template<class T, class T2>
	typename std::enable_if<std::is_arithmetic<T>::value && std::is_arithmetic<T2>::value, assign_events&>::type
	ranged(T&, const T2&, const T2&)
	{
		return *this;
	}

	template<class T, class T2, class Str, class... Args>
	typename std::enable_if<std::is_arithmetic<T>::value && std::is_arithmetic<T2>::value, assign_events&>::type
	ranged(T& target, const T2& min, const T2& max, Str&& name, Args&&... names)
	{
		if (event_name == name) {
			if (auto float_event = std::dynamic_pointer_cast<event::EventDouble>(event)) {
				if (float_event->range_specified()) {
					const auto& range_min = float_event->get_min_value();
					const auto& range_max = float_event->get_max_value();
					const auto& range_size = range_max - range_min;
					const auto& new_range = max - min;
					target = static_cast<T>(min + new_range*(float_event->get_value() - range_min)/range_size);
				} else {
					target = std::min(std::max(event::lex_cast_value<T>(event), static_cast<T>(min)),static_cast<T>(max));
				}
				assigned = true;
				return *this;
			} else if (auto int_event = std::dynamic_pointer_cast<event::EventInt>(event)) {
				if (float_event->range_specified()) {
					const auto& range_min = int_event->get_min_value();
					const auto& range_max = int_event->get_max_value();
					const auto& range_size = range_max - range_min;
					const auto& new_range = max - min;
					target = static_cast<T>(min + new_range*(int_event->get_value() - range_min)/range_size);
				} else {
					target = std::min(std::max(event::lex_cast_value<T>(event), static_cast<T>(min)),static_cast<T>(max));
				}
				assigned = true;
				return *this;
			}
		}
		return (*this)(target, names...);
	}
	operator bool() { return assigned; }
private:
	const std::string& event_name;
	const event::pBasicEvent& event;
	bool assigned;
};

}



#endif /* ASSIGN_EVENTS_H_ */