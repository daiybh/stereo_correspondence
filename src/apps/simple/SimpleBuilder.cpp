/*
 * SimpleBuilder.cpp
 *
 *  Created on: 11. 1. 2015
 *      Author: neneko
 */

#include "SimpleBuilder.h"
#include "yuri/core/thread/builder_utils.h"
#include <boost/regex.hpp>
#include <stdexcept>
#include <algorithm>
namespace yuri {

namespace simple {

namespace {

std::string gen_name(const std::string& type, const std::string& cls, int i)
{
	return type+"_"+cls+"_"+std::to_string(i);
}

/*!
 * Parses argument
 *
 * argument should be either just node name, or in the form:
 * name(param=value,param=value)
 *
 * @param log
 * @param arg
 * @param i
 * @return
 */
core::node_record_t parse_argument(log::Log& log, const std::string& arg, int i)
{
	core::node_record_t node;
	boost::regex node_expr ("([a-zA-Z0-9_-]+)(\\[[^]]+\\])?");
	boost::smatch what;
	auto start = arg.cbegin();
	const auto end = arg.cend();

	if (!regex_search(start, end, what, node_expr, boost::match_default)) {
		throw std::runtime_error("Failed to parse "+arg);
	}
	node.class_name = std::string(what[1].first, what[1].second);
	if (std::distance(what[2].first, what[2].second) > 2) {
//		boost::regex param_line("([^=]+)=([^,]+)(,)?");

		// supported param values:
		//				"abcahdkashdlka ,)(0 "
		//				abc
		//				abc(asd)()(ddd, 3, s)

		boost::regex param_line("([^=]+)=" // Parameter name
				"([^\",]"  "(([^,(]*(\\([^)]*\\))*)*)|" // Unqouted values (with optional brackets)
				"(\"[^\"]*\"))(,)?"); // Quoted values

		boost::sregex_iterator it(what[2].first+1, what[2].second-1, param_line, boost::match_default);
		boost::sregex_iterator it_end;
		while (it != it_end) {
			const auto& res = *it;
			const auto param_name  =  std::string(res[1].first,res[1].second);
			const auto param_value = std::string(std::string(res[2].first,res[2].second));
			++it;
			if (auto parsed_event = event::BasicEventParser::parse_expr(log, param_value, std::map<std::string,event::pBasicEvent>{})) {
				node.parameters[param_name]= parsed_event;
			} else {
				node.parameters[param_name] = param_value;
			}
		}
	}
	node.name = gen_name("node", node.class_name, i);
	return node;
}

}



SimpleBuilder::SimpleBuilder(const log::Log& log_, core::pwThreadBase parent, const std::vector<std::string>& argv)
:GenericBuilder(log_, parent,"simple")
{
	core::builder::load_builtin_modules(log);

	core::node_map nodes;
	core::link_map links;

	std::string last;
	int i = 0;
	for (const auto& s: argv) {
		auto node = parse_argument(log, s,++i);
		nodes[node.name]=node;//{name, node_cls, {}, {}};
		if (!last.empty()) {
			auto link_name = gen_name("link","single",i);
			log[log::info] << "link " << link_name << " from " << last << " to " << node.name;
			links[link_name]={link_name, "single", {}, last, node.name, 0, 0, {}};
		}

		last = node.name;
	}
	set_graph(nodes, links);
}


}
}


