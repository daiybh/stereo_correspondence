/*
 * ApplicationBuilder.h
 *
 *  Created on: Jul 28, 2010
 *      Author: worker
 */

#ifndef APPLICATIONBUILDER_H_
#define APPLICATIONBUILDER_H_
//#include "yuri/yuriconf.h"
#include "yuri/io/BasicIOThread.h"
#include "yuri/config/Config.h"
#include "yuri/config/Parameters.h"
#include "yuri/config/RegisteredClass.h"
#define TIXML_USE_STL
#include "tinyxml/tinyxml.h"
#include <map>

namespace yuri {

namespace config {

using namespace yuri::io;
using namespace yuri::threads;
using namespace boost::posix_time;
struct NodeRecord {
std::string name, type;
	Parameters params;
	std::map<std::string, std::string> variables;
	NodeRecord(std::string name, std::string type):name(name),type(type) {}
};

struct LinkRecord {
std::string name, source_node, target_node;
	int source_index, target_index;
	Parameters params;
	std::map<std::string, std::string> variables;
	LinkRecord(std::string name, std::string source_node,std::string target_node,
			int source_index, int target_index):
		name(name), source_node(source_node), target_node(target_node),
		source_index(source_index), target_index(target_index) {}
};

struct VariableNodeDependency{
	weak_ptr<NodeRecord> node;
	std::string parameter;
};
struct VariableLinkDependency{
	weak_ptr<LinkRecord> node;
	std::string parameter;
};
struct VariableRecord {
	std::string name, def, value;
	VariableRecord(std::string name, std::string def):name(name),def(def),value(def) {}
	std::vector<shared_ptr<VariableNodeDependency> > node_dependencies;
	std::vector<shared_ptr<VariableLinkDependency> > linkdependencies;
};

class EXPORT ApplicationBuilder: public BasicIOThread {
public:
	ApplicationBuilder(Log &_log, pThreadBase parent,std::string filename="", std::vector<std::string> argv=std::vector<std::string>());
	ApplicationBuilder(Log &_log, pThreadBase parent, Parameters &params);
	virtual ~ApplicationBuilder();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters);
	static shared_ptr<Parameters> configure();

	bool load_file(std::string path);
	void run();
	shared_ptr<BasicIOThread> get_node (std::string id);
	bool prepare_threads();
	bool find_modules();
	bool load_modules();
protected:
	bool process_module_dir(TiXmlElement &element);
	bool process_module(TiXmlElement &element);
	bool process_node(TiXmlElement &element);
	bool process_link(TiXmlElement &element);
	bool process_variable(TiXmlElement &element);
	bool parse_parameters(TiXmlElement &element,Parameters &params);
	void clear_tree();
	//bool check_links();


	bool prepare_links();
	bool spawn_threads();
	bool stop_threads();
	bool delete_threads();
	bool delete_pipes();
	bool step();

	void init();
	void init_local_params();
	void show_params(Parameters& _params,std::string prefix="\t\t");
	void fetch_tids();
	shared_ptr<Parameters> assign_variables(std::map<std::string, std::string> vars);
	void parse_argv(std::vector<std::string> argv);
protected:
std::string filename;
	TiXmlDocument doc;

	bool document_loaded, threads_prepared;
	boost::posix_time::time_duration run_limit;
	ptime start_time;
	shared_ptr<Parameters> default_pipe_param;
	std::map<std::string,shared_ptr<NodeRecord> > nodes;
	std::map<std::string,shared_ptr<LinkRecord> > links;
	std::map<std::string,shared_ptr<BasicIOThread> > threads;
	std::map<std::string,shared_ptr<BasicPipe> > pipes;
	std::map<std::string,shared_ptr<VariableRecord> > variables;
	std::vector<std::string> modules;
	std::vector<std::string> module_dirs;
	std::map<std::string,pid_t > tids;

};

}

}

#endif /* APPLICATIONBUILDER_H_ */
