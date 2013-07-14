/*!
 * @file 		EventHelpers.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		13.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef EVENTHELPERS_H_
#define EVENTHELPERS_H_
#include "BasicEvent.h"
namespace yuri {
namespace event {
class bad_event_cast: public std::runtime_error
{
public:
								bad_event_cast(const std::string& reason)
									:std::runtime_error(reason){}
};

typedef std::pair<event_type_t, event_type_t> event_pair;

template<class EventType>
typename event_traits<EventType>::stored_type
					get_value(const pBasicEvent& event) {
							const auto& event_ = dynamic_pointer_cast<EventType>(event);
							if (!event_) throw bad_event_cast("Type mismatch");
							return event_->get_value();
					}

template<class Src, class Dest>
pBasicEvent 		s_cast(const pBasicEvent& src) {
						const std::shared_ptr<Src>& ev = dynamic_pointer_cast<Src> (src);
						if (!ev) throw bad_event_cast("Static cast failed"); // This should actually never happen
						return std::make_shared<Dest>(static_cast<typename Dest::stored_type>(ev->get_value()));
					}
template<class Src, class Dest>
pBasicEvent 		lex_cast(const pBasicEvent& src) {
						const std::shared_ptr<Src>& ev = dynamic_pointer_cast<Src> (src);
						if (!ev) throw bad_event_cast("Static cast failed"); // This should actually never happen
						return std::make_shared<Dest>(lexical_cast<typename Dest::stored_type>(ev->get_value()));
					}

inline bool 		cmp_event_pair(const event_pair& a, const event_pair& b)
					{
						return a == b;
					}



}
}


#endif /* EVENTHELPERS_H_ */
