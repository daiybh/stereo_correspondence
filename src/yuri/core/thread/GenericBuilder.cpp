/*!
 * @file 		GenericBuilder.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		9.11.2014
 * @copyright	CESNET, z.s.p.o, 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "GenericBuilder.h"
#include "yuri/core/thread/IOThreadGenerator.h"
#include "yuri/core/pipe/PipeGenerator.h"
#include "yuri/core/utils/irange.h"
namespace yuri {
namespace core {

namespace {
	const std::string target_builder {"@"};
}

bool is_special_link_target(const std::string& name)
{
	return name == target_builder;
}

GenericBuilder::GenericBuilder(const log::Log& log_, pwThreadBase parent, const std::string& name)
:IOThread(log_, parent, 0, 0, name),BasicEventParser(log)
{

}


void GenericBuilder::run()
{
	if (!prepare_nodes()) return;
	if (!start_links()) return;
	if (!prepare_routing()) return;
	if (!start_nodes()) return;
	IOThread::run();
}

void GenericBuilder::set_graph(node_map nodes, link_map links, std::string routing)
{
	nodes_ = std::move(nodes);
	links_ = std::move(links);
	routing_=std::move(routing);
}

bool GenericBuilder::prepare_nodes()
{
	for (auto& node: nodes_) {
		auto& record = node.second;
		const auto& name = record.name;
		const auto& class_name = record.class_name;
		Parameters params = IOThreadGenerator::get_instance().configure(class_name);
		params.merge(record.parameters);
		params["_node_name"]=name;
		if (!(record.instance = IOThreadGenerator::get_instance().generate(class_name, log, get_this_ptr(), params))) {
			return false;
		}
		log[log::debug] << "Node " << name << " created successfully";
	}
	return true;
}


bool GenericBuilder::start_links()
{
	for (auto& link: links_) {
		auto& record = link.second;
		const auto& name = record.name;
		const auto& class_name = record.class_name;
		Parameters params = PipeGenerator::get_instance().configure(class_name);
		if (!(record.pipe = PipeGenerator::get_instance().generate(class_name, name, log, params.merge(record.parameters)))) {
			return false;
		}
		pIOThread source = get_node(record.source_node);
		if (!source) {
			log[log::error] << "Source node '" << record.source_node << "' for link " << name;
			return false;
		}
		pIOThread target = get_node(record.target_node);
		if (!target) {
			log[log::error] << "Target node '" << record.target_node << "' for link " << name;
			return false;
		}
		try {
			source->connect_out(record.source_index, record.pipe);
		}
		catch (std::out_of_range& ) {
			log[log::error] << "Output pipe index out of range: " << record.source_node << ":" << record.source_index;
			return false;
		}
		try {
			target->connect_in(record.target_index, record.pipe);
		}
		catch (std::out_of_range& ) {
			log[log::error] << "Input pipe index out of range: " << record.target_node << ":" << record.target_index;
			return false;
		}
		log[log::debug] << "Pipe " << name << " created successfully";
	}
	return true;
}
bool GenericBuilder::prepare_routing()
{
	if (!routing_.empty() && !parse_routes(routing_)) log[log::warning] << "Failed to parse routes ("<<routing_<<")";
	return true;
}

bool GenericBuilder::start_nodes()
{
	for (auto& node: nodes_) {
		if (!spawn_thread(node.second.instance)) return false;
	}
	return true;
}

pIOThread GenericBuilder::get_node(const std::string& name)
{
	pIOThread p;
	auto it = nodes_.find(name);
	if (it != nodes_.end()) {
		p = it->second.instance;
		if (p) log[log::debug] << "Found " << name;
	}
	if (!p) {
		if (/*name == this->name ||*/ name == "@") {
			p = std::dynamic_pointer_cast<IOThread>(get_this_ptr());
		}
		if (p) log[log::debug] << "Resolved " << name << " as this";
	}
	return p;
}

event::pBasicEventProducer GenericBuilder::find_producer(const std::string& name)
{
	event::pBasicEventProducer p;
	if (pIOThread node = get_node(name)) {
		p = std::dynamic_pointer_cast<event::BasicEventProducer>(node);
	}
	return p;
}
event::pBasicEventConsumer GenericBuilder::find_consumer(const std::string& name)
{
	event::pBasicEventConsumer p;
	if (pIOThread node = get_node(name)) {
		p = std::dynamic_pointer_cast<event::BasicEventConsumer>(node);
	}
	return p;
}

bool GenericBuilder::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (event_name == "stop") {
		log[log::info] << "Received stop event. Quitting builder.";
		request_end(yuri_exit_interrupted);
	}
	emit_event(event_name, event);
	return BasicEventParser::do_process_event(event_name, event);
}


bool GenericBuilder::step()
{
	process_events();
	run_routers();
	for (auto i: irange(0,get_no_in_ports())) {
		while (auto frame = pop_frame(i)) {
			push_frame(i, frame);
		}
	}
	return true;
}

void GenericBuilder::do_connect_in(position_t position, pPipe pipe)
{
	if (position >= do_get_no_in_ports()) {
		resize(position+1, -1);
	}
	return IOThread::do_connect_in(position, pipe);
}
void GenericBuilder::do_connect_out(position_t position, pPipe pipe)
{
	if (position >= do_get_no_out_ports()) {
		resize(-1, position+1);
	}
	return IOThread::do_connect_out(position, pipe);
}

void GenericBuilder::receive_event_hook() noexcept
{
	notify();
}

}
}
