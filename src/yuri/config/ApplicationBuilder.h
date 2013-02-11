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
using namespace std;
using namespace yuri::io;
using namespace yuri::threads;
using namespace boost::posix_time;
struct NodeRecord {
	string name, type;
	Parameters params;
	map<string,string> variables;
	NodeRecord(string name,string type):name(name),type(type) {}
};

struct LinkRecord {
	string name, source_node, target_node;
	int source_index, target_index;
	Parameters params;
	map<string,string> variables;
	LinkRecord(string name,string source_node, string target_node,
			int source_index, int target_index):
		name(name), source_node(source_node), target_node(target_node),
		source_index(source_index), target_index(target_index) {}
};

struct VariableNodeDependency{
	weak_ptr<NodeRecord> node;
	string parameter;
};
struct VariableLinkDependency{
	weak_ptr<LinkRecord> node;
	string parameter;
};
struct VariableRecord {
	string name, def, value;
	VariableRecord(string name,string def):name(name),def(def),value(def) {}
	vector<shared_ptr<VariableNodeDependency> > node_dependencies;
	vector<shared_ptr<VariableLinkDependency> > linkdependencies;
};

class ApplicationBuilder: public BasicIOThread {
public:
	EXPORT ApplicationBuilder(Log &_log, pThreadBase parent, string filename="", vector<string> argv=vector<string>()) throw (InitializationFailed);
	EXPORT ApplicationBuilder(Log &_log, pThreadBase parent, Parameters &params) throw (InitializationFailed);
	virtual ~ApplicationBuilder();
	EXPORT static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	EXPORT static shared_ptr<Parameters> configure();

	EXPORT bool load_file(string path);
	EXPORT void run();
	EXPORT shared_ptr<BasicIOThread> get_node(string id);
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
	void show_params(Parameters& _params, string prefix="\t\t");
	void fetch_tids();
	shared_ptr<Parameters> assign_variables(map<string,string> vars);
	void parse_argv(vector<string> argv);
protected:
	string filename;
	TiXmlDocument doc;

	bool document_loaded, threads_prepared;
	boost::posix_time::time_duration run_limit;
	ptime start_time;
	shared_ptr<Parameters> default_pipe_param;
	map<string,shared_ptr<NodeRecord> > nodes;
	map<string,shared_ptr<LinkRecord> > links;
	map<string,shared_ptr<BasicIOThread> > threads;
	map<string,shared_ptr<BasicPipe> > pipes;
	map<string,shared_ptr<VariableRecord> > variables;
	std::vector<std::string> modules;
	std::vector<std::string> module_dirs;
	map<string,pid_t > tids;

};

}

}

#endif /* APPLICATIONBUILDER_H_ */
