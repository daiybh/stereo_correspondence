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
								BasicEventProducer() = default;
	virtual						~BasicEventProducer();
	bool 						register_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name);
	bool 						unregister_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name);
	std::vector<event_info_t>	list_events();
protected:
	bool 						emit_event(const std::string& event_name, pBasicEvent event);
private:
	bool 						do_register_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name);
	bool 						do_unregister_listener(const std::string& event_name, pwBasicEventConsumer consumer, const std::string& target_name);
	virtual	std::vector<event_info_t>
								do_list_events() { return {}; }
	virtual bool				verify_register_event(const::std::string& event_name);
	std::unordered_multimap<std::string, event_target_t>
								consumers_;
	mutex						consumers_mutex_;

};


}
}

#endif /* BASICEVENTPRODUCER_H_ */
