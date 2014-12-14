/*!
 * @file 		XmlBuilder.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		15.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "XmlBuilder.h"
#include "yuri/core/thread/IOThreadGenerator.h"
#include "yuri/core/pipe/PipeGenerator.h"
#include "yuri/core/utils/ModuleLoader.h"
#include "builder_utils.h"
#define TIXML_USE_STL
#include "yuri/core/tinyxml/tinyxml.h"
#ifdef YURI_CYGWIN
#include <cstdlib>
#endif
namespace yuri {
namespace core {


IOTHREAD_GENERATOR(XmlBuilder)

MODULE_REGISTRATION_BEGIN("xml_builder")
		REGISTER_IOTHREAD("xml_builder", XmlBuilder)
MODULE_REGISTRATION_END()

namespace {

const std::string root_tag 			{"app"};
const std::string module_tag 		{"module"};
const std::string module_dir_tag 	{"module_dir"};
const std::string variable_tag 		{"variable"};
const std::string general_tag 		{"general"};
const std::string parameter_tag		{"parameter"};
const std::string node_tag 			{"node"};
const std::string link_tag 			{"link"};
const std::string description_tag 	{"description"};
const std::string event_tag			{"event"};

const std::string path_attrib		{"path"};
const std::string name_attrib		{"name"};
const std::string class_attrib		{"class"};
const std::string description_attrib{"description"};
const std::string source_attrib		{"source"};
const std::string target_attrib		{"target"};


template<typename T, typename U>
typename std::enable_if<std::is_convertible<U, T>::value,bool>::type
load_value(TiXmlElement* element, const std::string& name, T& value, const U& def) {
	if (element->QueryValueAttribute(name, &value) == TIXML_SUCCESS) return true;
	value = def;
	return true;
}


#ifdef YURI_CYGWIN
// For some reason, there's no std::stoll under cygwin...
	long long stoll(const std::string& str) { return std::atol(str.c_str()); }
#else
	using std::stoll;
#endif

}
struct XmlBuilder::builder_pimpl_t{
	builder_pimpl_t(log::Log& log_, XmlBuilder& builder):log(log_),builder(builder){}
	void load_file(const std::string&);

	void process_modules();
	void process_module_dirs();
	void process_argv(const std::vector<std::string>&);
	void process_variables();
	Parameters process_general();
	void process_nodes();
	void process_links();

	void process_routing();

	void load_builtin_modules();

	event::pBasicEvent parse_expression(const std::string expression);
	Parameters parse_parameters(const TiXmlElement* element);

	void verify_node_class(const std::string& class_name);
	void verify_link_class(const std::string& class_name);
	void verify_links();

	pIOThread get_node(const std::string& name);
	log::Log& log;
	XmlBuilder& builder;
	std::unique_ptr<TiXmlDocument> doc;
	TiXmlElement* root {nullptr};
	std::string name;
	std::string description;
//	std::vector<std::string> routing_info;


	std::vector<std::string> module_dirs;
	Parameters argv;
	Parameters variables;
	std::map<std::string, event::pBasicEvent> input_events;
	std::map<std::string, node_record_t> nodes;
	std::map<std::string, link_record_t> links;
	std::string routing;
};

#define VALID(x,msg) if (!(x)) { log[log::error] << msg; throw exception::InitializationFailed(msg); }
#define VALID_XML(x) VALID(x,"Invalid XML")

void XmlBuilder::builder_pimpl_t::load_file(const std::string& file)
{

	doc.reset(new TiXmlDocument());
	VALID(doc->LoadFile(file),"Failed to load file " + file);
	root = doc->RootElement();
	VALID_XML(root)
	load_value(root, name_attrib, name, "yuri2.8");
	TiXmlElement * node = nullptr;
	if((node = dynamic_cast<TiXmlElement*>(root->IterateChildren(description_tag, node)))) {
		const char* text = node->GetText(); // GetText may return nullptr and basic_string::operator= doesn't accept nullptr...
		if (text) description = text;
	}
	process_routing();
}

void XmlBuilder::builder_pimpl_t::process_modules()
{
//	log[log::verbose_debug] << "Loading <module>s";
	TiXmlElement * node = nullptr;
	std::vector<std::string> modules;
	VALID_XML(root);
	while((node = dynamic_cast<TiXmlElement*>(root->IterateChildren(module_tag, node)))) {
		std::string path;
		if (node->QueryValueAttribute(path_attrib, &path)!=TIXML_SUCCESS) continue;
		modules.push_back(std::move(path));
	}
	builder::load_modules(log, modules);
}

void XmlBuilder::builder_pimpl_t::process_module_dirs()
{
//	log[log::verbose_debug] << "Loading <module_dir>s";
	TiXmlElement * node = nullptr;
	std::vector<std::string> module_dirs;
	while((node = dynamic_cast<TiXmlElement*>(root->IterateChildren(module_dir_tag, node)))) {
		std::string path;
		if (node->QueryValueAttribute(path_attrib, &path)!=TIXML_SUCCESS) continue;
		module_dirs.push_back(std::move(path));
	}
	for (const auto& m: module_dirs) {
		builder::load_module_dir(log, m);
	}
}

void XmlBuilder::builder_pimpl_t::load_builtin_modules()
{
	builder::load_builtin_modules(log);
}

void XmlBuilder::builder_pimpl_t::process_argv(const std::vector<std::string>& var)
{
	for (const std::string& param_pair: var) {
//		log[log::info] << "Checking " << param_pair;
		auto idx = param_pair.find('=');
//		log[log::info] << "idx: " << idx;
		if (idx == param_pair.npos) continue;
		const std::string& val = param_pair.substr(idx+1);
		Parameter p(param_pair.substr(0,idx));
		if (event::pBasicEvent event = event::BasicEventParser::parse_expr(log, val,{})) {
//			log[log::info] << "Got event" ;
			p.set_value(event);
		} else {
//			log[log::info] << "No event, assuming a string" ;
			p=val;
		}
		argv[p.get_name()]=std::move(p);
	}
//	for (const auto&p: argv) {
//		log[log::info] << "Found parameter " << p.first << ": " << p.second.get<std::string>();
//	}
}
void XmlBuilder::builder_pimpl_t::process_variables()
{
	TiXmlElement * node = nullptr;
	for (const auto& av: argv) {
		input_events[av.first]=av.second.get_value();
	}
	while((node = dynamic_cast<TiXmlElement*>(root->IterateChildren(variable_tag, node)))) {
		std::string name;
		if (node->QueryValueAttribute(name_attrib, &name)!=TIXML_SUCCESS) continue;
//		log[log::info] << "Checking variable " << name;
		std::string desc;
		node->QueryValueAttribute(description_attrib, &desc);// This is optional
		auto it = input_events.find(name);
		if (it != input_events.end()){
			variables[name][desc].set_value(it->second);
		} else {
			const char* text = node->GetText();
			if (!text) text = "";
			variables[name][desc]=parse_expression(text);
			input_events[name]=variables[name].get_value();
		}
	}
//	for (const auto&p: variables) {
//		log[log::info] << "Found variable: " << p.first << ": " << p.second.get<std::string>();
//	}
}
Parameters XmlBuilder::builder_pimpl_t::process_general()
{
	TiXmlElement* node = dynamic_cast<TiXmlElement*>(root->IterateChildren(general_tag, nullptr));
	return parse_parameters(node);
}

event::pBasicEvent XmlBuilder::builder_pimpl_t::parse_expression(const std::string expression)
{
	if (event::pBasicEvent event = event::BasicEventParser::parse_expr(log, expression, input_events)){
//		log[log::info] << "Got event" ;
		return event;
//		p.set_value(event);
	} else {
//		log[log::info] << "No event, assuming a string" ;
		return make_shared<event::EventString>(expression);
	}
}
Parameters XmlBuilder::builder_pimpl_t::parse_parameters(const TiXmlElement* element)
{
	Parameters params;
	if (element) {
		const TiXmlElement * node = nullptr;
		while((node = dynamic_cast<const TiXmlElement*>(element->IterateChildren(parameter_tag, node)))) {
			std::string name;
			if (node->QueryValueAttribute(name_attrib, &name)!=TIXML_SUCCESS) continue;
			const char* text = node->GetText();
			if (!text) text = "";
			params[name]=parse_expression(text);
		}
	}
	return params;
}
void XmlBuilder::builder_pimpl_t::process_nodes()
{
	TiXmlElement * node = nullptr;
	while((node = dynamic_cast<TiXmlElement*>(root->IterateChildren(node_tag, node)))) {
		node_record_t record;
		VALID_XML(node->QueryValueAttribute(name_attrib, &record.name)==TIXML_SUCCESS)
		VALID_XML(node->QueryValueAttribute(class_attrib, &record.class_name)==TIXML_SUCCESS)
		record.parameters = parse_parameters(node);
		builder::verify_node_class(record.class_name);
		nodes[record.name]=std::move(record);
	}
}
void XmlBuilder::builder_pimpl_t::process_links()
{
	TiXmlElement * node = nullptr;
	while((node = dynamic_cast<TiXmlElement*>(root->IterateChildren(link_tag, node)))) {
		link_record_t record;
		VALID_XML(node->QueryValueAttribute(name_attrib, &record.name)==TIXML_SUCCESS)
		VALID_XML(node->QueryValueAttribute(class_attrib, &record.class_name)==TIXML_SUCCESS)
		std::string source, target;
		VALID_XML(node->QueryValueAttribute(source_attrib, &source)==TIXML_SUCCESS)
		VALID_XML(node->QueryValueAttribute(target_attrib, &target)==TIXML_SUCCESS)
		auto idx = source.find(':');
		VALID(idx!=std::string::npos,"Malformed source specification in source for link " + record.name)
		record.source_node = source.substr(0,idx);
		record.source_index = stoll(source.substr(idx+1));
		idx = target.find(':');
		VALID(idx!=std::string::npos,"Malformed target specification in source for link " + record.name)
		record.target_node = target.substr(0,idx);
		record.target_index = stoll(target.substr(idx+1));
		record.parameters = parse_parameters(node);
		builder::verify_link_class(record.class_name);
		links[record.name]=std::move(record);
	}
}
void XmlBuilder::builder_pimpl_t::process_routing()
{
	TiXmlElement * node = nullptr;
	if((node = dynamic_cast<TiXmlElement*>(root->IterateChildren(event_tag, node)))) {
		const char* text = node->GetText();
		//if (text) routing_info.push_back(text);
		if (text) routing.append(text);
	}
}

void XmlBuilder::builder_pimpl_t::verify_links()
{
	using map_elem = std::pair<std::string, position_t>;
	std::map<map_elem, std::string> used_sources, used_targets;
	for (const auto& link: links) {
		auto s = nodes.find(link.second.source_node);
		auto t = nodes.find(link.second.target_node);
		VALID(s != nodes.end(),"Unknown source node " + link.second.source_node + " in link " +link.first)
		VALID(t != nodes.end(),"Unknown target node " + link.second.target_node + " in link " +link.first)
		map_elem selem {link.second.source_node, link.second.source_index};
		map_elem telem {link.second.target_node, link.second.target_index};
		auto s2 = used_sources.find(selem);
		auto t2 = used_targets.find(telem);
		VALID(s2 == used_sources.end() || link.second.source_index < 0, "Duplicate specification for source " + selem.first +":"+lexical_cast<std::string>(selem.second)+", specified in "+s2->second+" and "+link.first)
		VALID(t2 == used_targets.end() || link.second.target_index < 0, "Duplicate specification for target " + telem.first +":"+lexical_cast<std::string>(telem.second)+", specified in "+t2->second+" and "+link.first)
		used_sources[selem]=link.first;
		used_targets[telem]=link.first;
	}
}

#undef VALID
#undef VALID_XML

Parameters XmlBuilder::configure()
{
	Parameters p = IOThread::configure();
	p["filename"]["Path to  XML file."]="";
	return p;
}


XmlBuilder::XmlBuilder(const log::Log& log_, pwThreadBase parent, const Parameters& parameters)
:GenericBuilder(log_,parent,"XmlBuilder")
{
	pimpl_.reset(new builder_pimpl_t(log,*this));
	set_latency(10_ms);
//	pimpl_->load_file(filename_);
	pimpl_->load_file(parameters["filename"].get<std::string>());

	pimpl_->process_modules();
	pimpl_->process_module_dirs();
//	pimpl_->process_argv(argv);
	pimpl_->process_variables();
	Parameters general = pimpl_->process_general();
	IOTHREAD_INIT(general.merge(parameters)); //TODO Shouldn't this be the other way round? general won't be used at all if used like this...
	pimpl_->load_builtin_modules();
	pimpl_->process_nodes(); // TODO process all nodes
	pimpl_->process_links(); // TODO process all links
	pimpl_->verify_links();
	log[log::info] << "File seems to be parsed successfully";
	set_graph(pimpl_->nodes, pimpl_->links, pimpl_->routing);

}
XmlBuilder::XmlBuilder(const log::Log& log_, pwThreadBase parent, const std::string& filename, const std::vector<std::string>& argv, bool parse_only)
:GenericBuilder(log_,parent,"XmlBuilder")
{
	pimpl_.reset(new builder_pimpl_t(log, *this));
	set_latency(10_ms);
	try {
		filename_ = filename;
		pimpl_->load_file(filename_);
		pimpl_->process_modules();
		pimpl_->process_module_dirs();
		pimpl_->process_argv(argv);
		pimpl_->process_variables();
		Parameters general = pimpl_->process_general();
		IOTHREAD_INIT(general);
	} catch (...) {
		if (!parse_only) throw;
	}
	pimpl_->load_builtin_modules();

	if (parse_only) return;
	pimpl_->process_nodes(); // TODO process all nodes
	pimpl_->process_links(); // TODO process all links
	pimpl_->verify_links();
	log[log::info] << "File seems to be parsed successfully";
	set_graph(pimpl_->nodes, pimpl_->links, pimpl_->routing);
}

XmlBuilder::~XmlBuilder() noexcept {}


bool XmlBuilder::set_param(const Parameter& parameter)
{
	if (parameter.get_name() == "filename") {
		filename_ = parameter.get<std::string>();
	} else return IOThread::set_param(parameter);
	return true;
}

const std::string& XmlBuilder::get_app_name()
{
	return pimpl_->name;
}
const std::string& XmlBuilder::get_description()
{
	return pimpl_->description;
}
std::vector<variable_info_t> XmlBuilder::get_variables() const
{
	std::vector<variable_info_t> vars;
	for (const auto& v: pimpl_->variables) {
		const auto& var = v.second;
		vars.push_back({var.get_name(), var.get_description(), var.get<std::string>()});
	}
	return vars;
}
}
}

