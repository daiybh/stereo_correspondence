/*!
 * @file 		BasicEventParser.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		09.07.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef BASICEVENTPARSER_H_
#define BASICEVENTPARSER_H_
#include "BasicEventProducer.h"

namespace yuri {
namespace event {
namespace parser {
enum class 	token_type_t {
	invalid,
	route,
	brace,
	func_name,
	spec,
	int_const,
	double_const,
	string_const,
	bool_const,
	vector_const,
	dict_const
};

enum class func_mode_t {
	evaluate_all,
	evaluate_first
};
typedef shared_ptr<struct token>	p_token;
struct token {
	token_type_t 				type;
								token(token_type_t tok):type(tok) {}
	virtual 					~token() {}
};
struct invalid_token: public token {
								invalid_token():token(token_type_t::invalid) {}

};
struct route_token: public token {
								route_token():token(token_type_t::route) {}
	p_token 					expr;
	std::vector<p_token> 		output;
};
struct spec_token: public token {
								spec_token(const std::string& node, const std::string& name)
									:token(token_type_t::spec),node(node),name(name) {}
	const std::string 			node;
	const std::string 			name;
};
struct func_token: public token {
								func_token(const std::string& fname)
									:token(token_type_t::func_name),fname(fname),mode(func_mode_t::evaluate_all) {}
	std::string 				fname;
	func_mode_t 				mode;
	std::vector<p_token> 		args;
};
struct bool_const_token: public token {
								bool_const_token(bool val)
									:token(token_type_t::bool_const),val(val) {}
	bool 						val;
};
struct int_const_token: public token {
								int_const_token(int64_t val)
									:token(token_type_t::int_const),val(val) {}
	int64_t 					val;
};
struct double_const_token: public token {
								double_const_token(long double val)
									:token(token_type_t::double_const),val(val) {}
	long double 				val;
};
struct string_const_token: public token {
								string_const_token(const std::string& val)
									:token(token_type_t::string_const),val(val) {}
	std::string 				val;
};
struct vector_const_token: public token {
								vector_const_token()
									:token(token_type_t::vector_const) {}
	std::vector<p_token> 		members;
};
struct dict_const_token: public token {
								dict_const_token()
									:token(token_type_t::dict_const) {}
	std::map<std::string, p_token>
								members;
};


bool 							is_simple_route(const p_token& ast);
std::pair<std::vector<p_token>, std::string>
								parse_string(const std::string& text);



}
class EventRouter;
class BasicEventParser: public BasicEventProducer, public BasicEventConsumer {
public:
								BasicEventParser();
	virtual 					~BasicEventParser() {}
private:
	virtual pBasicEventProducer find_producer(const std::string& name) = 0;
	virtual pBasicEventConsumer find_consumer(const std::string& name) = 0;
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event);
protected:
	bool 						parse_routes(const std::string& text);
	bool 						run_routers();
private:
	std::vector<shared_ptr<EventRouter>>
								routers_;

};

}
}

#endif /* BASICEVENTPARSER_H_ */
