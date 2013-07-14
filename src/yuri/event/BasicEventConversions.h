/*!
 * @file 		BasicEventConversions.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		04.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef BASICEVENTCONVERSIONS_H_
#define BASICEVENTCONVERSIONS_H_
#include "BasicEvent.h"
#include "EventHelpers.h"
#include <unordered_map>
#include <functional>
namespace yuri {
namespace event {

typedef std::function<pBasicEvent(const std::vector<pBasicEvent>&)>
								event_function_t;

struct 							event_function_record_t
{
	const std::string fname;
	std::vector<event_type_t> param_types;
	event_type_t return_type;
	event_function_t func_;
};

class EventFunctions
{
public:
	void 						add_function(const event_function_record_t& frec);
	void 						add_functions(const std::vector<event_function_record_t>& records);
	std::unordered_multimap<std::string, event_function_record_t> &
								get_map()
									{ return functions_; };
								EventFunctions() {}
private:
	std::unordered_multimap<std::string, event_function_record_t>
								functions_;
	mutex						map_mutex_;
};

typedef  SingletonBase<EventFunctions>
								EventFunctionsFactory;


pBasicEvent 					implicit_conversion(pBasicEvent event, event_type_t target_type);

pBasicEvent 					call(const std::string& fname, const std::vector<pBasicEvent>& events);

template<class... Args>
pBasicEvent 					call(const std::string& fname, pBasicEvent event0, Args... args)
{
	return call(fname,{event0, args...});
}





}
}


#endif /* BASICEVENTCONVERSIONS_H_ */
