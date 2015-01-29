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
	operator bool() { return assigned; }
private:
	const std::string& event_name;
	const event::pBasicEvent& event;
	bool assigned;
};

}



#endif /* ASSIGN_EVENTS_H_ */
