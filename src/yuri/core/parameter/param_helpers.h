/*!
 * @file 		param_helpers.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		11.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef PARAM_HELPERS_H_
#define PARAM_HELPERS_H_
namespace yuri {
namespace core {

namespace detail {
template<typename T, class X = void> struct suitable_event_type;

template<typename T, typename Event> struct suitable_event_helper{
	using arg_type = T;
	using value_type = Event;
	using shared_type = shared_ptr<value_type>;
	static shared_type create_new(const arg_type& arg/*, bool use_cast = cast*/) {
		/*if (use_cast)*/ return make_shared<value_type>(lexical_cast<typename event::event_traits<value_type>::stored_type>(arg));
//		return make_shared<value_type>(arg);
	}


};

template<>
struct suitable_event_type<bool>
:public suitable_event_helper<bool, event::EventBool>{};

//template<>
//struct suitable_event_type<const char*>
//:public suitable_event_helper<const char*, event::EventString>{};
//
//template<>
//struct suitable_event_type<std::string>
//:public suitable_event_helper<std::string, event::EventString>{};

template<typename T>
struct suitable_event_type<T, typename std::enable_if<std::is_integral<T>::value>::type>
:public suitable_event_helper<T, event::EventInt> {};

template<typename T>
struct suitable_event_type<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
:public suitable_event_helper<T, event::EventDouble>{};

template<typename T>
struct suitable_event_type<T, typename std::enable_if<!std::is_integral<T>::value && !std::is_floating_point<T>::value>::type>
:public suitable_event_helper<T, event::EventString>{};

}
}
}




#endif /* PARAM_HELPERS_H_ */
