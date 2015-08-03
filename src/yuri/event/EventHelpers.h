/*!
 * @file 		EventHelpers.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		13.07.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef EVENTHELPERS_H_
#define EVENTHELPERS_H_
#include "BasicEvent.h"
#include "yuri/core/utils.h"
#ifdef YURI_WIN
#pragma warning ( push )
// Disable warning "Forcing value to bool ...
#pragma warning ( disable: 4800)
#endif
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
							const auto& event_ = std::dynamic_pointer_cast<EventType>(event);
							if (!event_) throw bad_event_cast("Type mismatch");
							return event_->get_value();
					}

template<class Src, class Dest>
pBasicEvent 		s_cast(const pBasicEvent& src) {
						const std::shared_ptr<Src>& ev = std::dynamic_pointer_cast<Src> (src);
						if (!ev) throw bad_event_cast("Static cast failed"); // This should actually never happen
						return std::make_shared<Dest>(static_cast<typename Dest::stored_type>(ev->get_value()));
					}
template<class Src, class Dest>
pBasicEvent 		lex_cast(const pBasicEvent& src) {
						const std::shared_ptr<Src>& ev = std::dynamic_pointer_cast<Src> (src);
						if (!ev) throw bad_event_cast("Static cast failed"); // This should actually never happen
						return std::make_shared<Dest>(lexical_cast<typename Dest::stored_type>(ev->get_value()));
					}

inline bool 		cmp_event_pair(const event_pair& a, const event_pair& b)
					{
						return a == b;
					}

// TODO: This is not scalable and should be redone... ideally by implementing operator>>(std::ostream&, duration_t);

template<typename T>
typename std::enable_if<std::is_same<T,duration_t>::value, duration_t>::type
					lex_cast_value(const pBasicEvent& src)
{
						const event_type_t type = src->get_type();
							switch (type) {
								case event_type_t::duration_event: return get_value<EventDuration>(src);
								default: throw bad_event_cast("Unsupported event type");
							}
}

template<typename T>
typename std::enable_if<!std::is_same<T,duration_t>::value, T>::type
 					lex_cast_value(const pBasicEvent& src) {
						const event_type_t type = src->get_type();
						switch (type) {
							case event_type_t::bang_event: throw bad_event_cast("No conversion for BANG values");
							case event_type_t::boolean_event: return lexical_cast<T>(get_value<EventBool>(src));
							case event_type_t::integer_event: return lexical_cast<T>(get_value<EventInt>(src));
							case event_type_t::double_event: return lexical_cast<T>(get_value<EventDouble>(src));
							case event_type_t::string_event: return lexical_cast<T>(get_value<EventString>(src));
							default: throw bad_event_cast("Unsupported event type");
						}
					}


}
}


#ifdef YURI_WIN
#pragma warning ( pop )
#endif
#endif /* EVENTHELPERS_H_ */
