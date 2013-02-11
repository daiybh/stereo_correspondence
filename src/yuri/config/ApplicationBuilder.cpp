/*
 * ApplicationBuilder.cpp
 *
 *  Created on: Jul 28, 2010
 *      Author: worker
 */

#include "ApplicationBuilder.h"
#include <boost/lexical_cast.hpp>
#include <cassert>

namespace yuri {

namespace config {

REGISTER("appbuilder",ApplicationBuilder)

shared_ptr<BasicIOThread> ApplicationBuilder::generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception)
{
	shared_ptr<ApplicationBuilder> app (new ApplicationBuilder(_log,parent,parameters));
	return app;
}
shared_ptr<Parameters> ApplicationBuilder::configure()
{
	shared_ptr<Parameters> p (new Parameters());
	(*p)["run_limit"]["Runtime limit in seconds"]=-1.0;
	(*p)["config"]["Config file"]="";
	(*p)["debug"]["Debug level (2 most verbose, -2 least verbose)"]="0";
	(*p)["show_time"]["show timestamps"]=true;
	(*p)["show_level"]["show debug levels"]=true;
	(*p)["use_colors"]["Use color output"]=true;
	return p;
}


ApplicationBuilder::ApplicationBuilder(Log &_log, pThreadBase parent, Parameters &p)
	throw (InitializationFailed):
	BasicIOThread(_log,parent,0,0,"AppBuilder"),filename(""),
	document_loaded(false),threads_prepared(false),
	run_limit(boost::posix_time::pos_infin),start_time(not_a_date_time)
{
	default_pipe_param=BasicPipe::configure();
	params.merge(p);
	init();
}

ApplicationBuilder::ApplicationBuilder(Log &_log, pThreadBase parent, string filename,vector<string> argv)
	throw (InitializationFailed):
	BasicIOThread(_log,parent,0,0,"AppBuilder"),filename(filename),document_loaded(false),
	threads_prepared(false),run_limit(boost::posix_time::pos_infin),
	start_time(not_a_date_time)

{
	default_pipe_param=BasicPipe::configure();
	shared_ptr<Parameters> p = configure();
	params.merge(*p);
	params["config"]=filename;
	init();
	parse_argv(argv);
}

ApplicationBuilder::~ApplicationBuilder()
{
	clear_tree();
#ifdef __linux__
	std::string destdir = string("/tmp/")+lexical_cast<std::string>(getpid());
	boost::filesystem::path p(destdir);
	if (boost::filesystem::exists(p)) {
		boost::filesystem::remove_all(p);
	}
#endif
}

void ApplicationBuilder::run()
{
	//check_links();
	double limit = params["run_limit"].get<double>();

	if (limit<0) run_limit=boost::posix_time::pos_infin;
	else {
		run_limit = boost::posix_time::seconds(floor(limit))+
			boost::posix_time::seconds(floor(1e6*(limit-floor(limit))));
	}
	log[debug] << "Got limit " << limit << ", which was converted to " << run_limit << endl;
	start_time=microsec_clock::local_time();
	try {
		if (!prepare_threads()) throw (InitializationFailed("Failed to prepare threads!"));
		log[debug] << "Threads prepared" << endl;
		if (!prepare_links()) throw (InitializationFailed("Failed to prepare links!"));
		log[debug] << "Links prepared" << endl;
		if (!spawn_threads()) throw (InitializationFailed("Failed to spawn threads!"));
		log[debug] << "Threads spawned" << endl;
		BasicIOThread::run();
	}
	catch (Exception &e) {
		log[error] << "Caught exception when initializing appbuilder ("
				<< e.what() << ")" << endl;
	}
	try {
		if (!stop_threads()) log[warning] << "Failed to stop threads!" << endl;
	}
	catch (Exception &e) {
		log[error] << "Caught exception while stopping threads ("
				<< e.what() << ")" << endl;
	}
	try {
		if (!delete_threads()) log[warning] << "Failed to delete threads!" << endl;
	}
	catch (Exception &e) {
		log[error] << "Caught exception while deleting threads ("
				<< e.what() << ")" << endl;
	}
	try {
		if (!delete_pipes()) log[warning] << "Failed to delete pipes!" << endl;
	}
	catch (Exception &e) {
		log[error] << "Caught exception while deleting pipes ("
				<< e.what() << ")" << endl;
	}
}
bool ApplicationBuilder::find_modules()
{
	BOOST_FOREACH(string p, module_dirs) {
		boost::filesystem::path p_(p);
		if (boost::filesystem::exists(p_) &&
				boost::filesystem::is_directory(p_)) {
			log[debug] << "Looking for modules in " << p << "\n";
			for (boost::filesystem::directory_iterator dit(p_);
					dit !=boost::filesystem::directory_iterator(); ++dit) {
				boost::filesystem::directory_entry d = *dit;
				const std::string name = d.path().filename().native();
				if (name.substr(0,16)=="libyuri2_module_") {
					log[info] << "\tFound module " << name << "\n";
					modules.push_back(d.path().native());
				} else log[info] << "" << name.substr(0,13) << "\n";
			}
		}
	}
	return true;
}
bool ApplicationBuilder::load_modules()
{
	BOOST_FOREACH(string s, modules) {
		try{
			bool loaded = RegisteredClass::load_module(s);
			log[info] << "Loading " << s<< ": "<<(loaded?"OK":"Failed") << "\n";
		}
		catch (yuri::exception::Exception &) {}

	}
	return true;
}
bool ApplicationBuilder::load_file(string path)
{
	//if (doc) delete doc;
	clear_tree();
	document_loaded = false;
	log[info] << "Loading " << path << endl;
	//doc.reset(new TiXmlDocument(path));
	if (! doc.LoadFile(path)) {
		log[error] <<  "Failed to load file! " << doc.ErrorDesc()<< endl;
		return false;
	}
	//log[debug] << "Loaded "<< endl;
	TiXmlElement * root = doc.RootElement();
	if (!root) {
		//doc.reset();;
		return false;
	}
	//log[debug] << "Got root element!" << endl;
	TiXmlElement* node=0;
	node = root->FirstChildElement("general");
	if (node) {
		parse_parameters(*node,params);
	}
	init_local_params();

	node=0;
	while ((node = dynamic_cast<TiXmlElement*>(root->IterateChildren("module_dir",node)))) {
		if (!process_module_dir(*node)) continue;
	}
	find_modules();
	node=0;
	while ((node = dynamic_cast<TiXmlElement*>(root->IterateChildren("module",node)))) {
		if (!process_module(*node)) continue;
	}

	load_modules();

	node=0;
	while ((node = dynamic_cast<TiXmlElement*>(root->IterateChildren("variable",node)))) {
		if (!process_variable(*node)) continue;
	}

	node=0;
	while ((node = dynamic_cast<TiXmlElement*>(root->IterateChildren("node",node)))) {
		if (!process_node(*node)) continue;
	}
	node = 0;
	while ((node = dynamic_cast<TiXmlElement*>(root->IterateChildren("link",node)))) {
		if (!process_link(*node)) continue;
	}
	return (document_loaded = true);
}
bool ApplicationBuilder::process_module_dir(TiXmlElement &node)
{
	string path;
	if (node.QueryValueAttribute("path",&path)!=TIXML_SUCCESS) {
		log[error] << "Failed to load path attribute!" << endl;
		return false;
	}
	log[debug] << "Found module dir" << path << "\n";
	module_dirs.push_back(path);
	return true;
}
bool ApplicationBuilder::process_module(TiXmlElement &node)
{
	string path;
	if (node.QueryValueAttribute("path",&path)!=TIXML_SUCCESS) {
		log[error] << "Failed to load path attribute!" << endl;
		return false;
	}
	log[debug] << "Found module " << path << "\n";
	modules.push_back(path);
	return true;
}


bool ApplicationBuilder::process_node(TiXmlElement &node)
{
	string name,cl;
	if (node.QueryValueAttribute("name",&name)!=TIXML_SUCCESS) {
		log[error] << "Failed to load name attribute!" << endl;
		return false;
	}
	if (node.QueryValueAttribute("class",&cl)!=TIXML_SUCCESS) {
		log[error] << "Failed to load class attribute!" << endl;
		return false;;
	}
	log[debug] << "Found Node " << name << " of class " << cl << endl;
	if (nodes.find(name) != nodes.end()) {
		log[warning] << "Multiple nodes with name " << name << \
				" specified! (was " << nodes[name]->type << ", found"
				<< cl << "). Discarding the the later" <<endl;
		return false;
	}

	if (!RegisteredClass::is_registered(cl)) {
		log[warning] << "Requested node of class " << cl
				<< " that is not registered! Discarding" << endl;
		return false;
	}
	shared_ptr<NodeRecord> n(new NodeRecord(name,cl));

	if (!parse_parameters(node,n->params)) {
		log[warning] << "Failed to parse parameters. Discarding" << endl;
		//delete n;
		return false;
	}
	pair<string,shared_ptr<Parameter> > parameter;
	BOOST_FOREACH(parameter,n->params.params) {
		string value = parameter.second->get<string>();
		if (value[0] == '@') {
			string argname = value.substr(1);
			n->variables[parameter.first]=argname;
		}
	}

	log[debug] << "Storing node " << name << endl;
	nodes[name]=n;
	return true;
}

bool ApplicationBuilder::process_link(TiXmlElement &node)
{
	string name,src,target;
	if (node.QueryValueAttribute("name",&name)!=TIXML_SUCCESS) {
		log[error] << "Failed to load name attribute!" << endl;
		return false;
	}
	if (node.QueryValueAttribute("source",&src)!=TIXML_SUCCESS) {
		log[error] << "Failed to load source attribute for link "
				<< name << "!" << endl;
		return false;
	}
	if (node.QueryValueAttribute("target",&target)!=TIXML_SUCCESS) {
		log[error] << "Failed to load target attribute for link "
				<< name << "!" << endl;
		return false;
	}
	int src_ipos = src.find_last_of(':');
	int target_ipos = target.find_last_of(':');
	string srcnode = src.substr(0,src_ipos);
	string targetnode = target.substr(0,target_ipos);
	int srci = boost::lexical_cast<int>(src.substr(src_ipos+1));
	int targeti = boost::lexical_cast<int>(target.substr(target_ipos+1));
	log[debug] << "link from " <<  srcnode << "[" << srci << "]" <<
			"to " << targetnode << "[" <<targeti << "]" << endl;
	if (nodes.find(srcnode)==nodes.end()) {
		log[warning] << "Source node for the link (" << srcnode
				<< ") not specified! Discarding link." << endl;
		return false;
	}
	if (nodes.find(targetnode)==nodes.end()) {
		log[warning] << "Target node for the link (" << targetnode
				<< ") not specified! Discarding link." << endl;
		return false;
	}
	shared_ptr<LinkRecord> l(new LinkRecord(name,srcnode,targetnode,srci,targeti));
	if (!parse_parameters(node,l->params)) {
		log[warning] << "Failed to parse parameters. Discarding" << endl;
		//delete l;
		return false;
	}
	pair<string,shared_ptr<Parameter> > parameter;
	BOOST_FOREACH(parameter,l->params.params) {
		string value = parameter.second->get<string>();
		if (value[0] == '@') {
			string name = value.substr(1);
			l->variables[parameter.first]=name;
		}
	}
	log[debug] << "Storing link " << name << endl;
	links[name]=l;
	return true;
}

bool ApplicationBuilder::process_variable(TiXmlElement &node)
{
	string name,def_value;
	if (node.QueryValueAttribute("name",&name)!=TIXML_SUCCESS) {
		log[error] << "Failed to load name attribute!" << endl;
		return false;
	}
	if (node.QueryValueAttribute("default",&def_value)!=TIXML_SUCCESS) {
		log[error] << "Failed to load default value for variable "
				<< name << "!" << endl;
		return false;
	}

	shared_ptr<VariableRecord> var(new VariableRecord(name,def_value));
	log[debug] << "Storing variable " << name << endl;
	variables[name]=var;
	return true;
}


bool ApplicationBuilder::parse_parameters(TiXmlElement &element,Parameters &params)
{
	TiXmlElement *node=0;
	while ((node = dynamic_cast<TiXmlElement*>(element.IterateChildren("parameter",node)))) {
		string name, value;
		if (node->QueryValueAttribute("name",&name)!=TIXML_SUCCESS) {
			log[error] << "Failed to load name attribute!" << endl;
			return false;
		}
		if (node->FirstChild("parameter")) {

			Parameters& par_child = params[name].push_group();
			log[info] << "Found group parameter " << name << ", added "
					<< params[name].parameters_vector.size() << ". value" << endl;
			if (!parse_parameters(*node,par_child)) return false;
		} else {
			value = node->GetText();
			log[debug] << "Found parameter " << name << " with value " << value << endl;
			params[name]=value;
		}
	}
	return true;
}

void ApplicationBuilder::clear_tree()
{
	/*pair<string, shared_ptr<NodeRecord> > n;
	BOOST_FOREACH(n,nodes) {
		delete n.second;
	}*/
	nodes.clear();
	/*pair<string, shared_ptr<LinkRecord> > l;
	BOOST_FOREACH(l,links) {
		delete l.second;
	}*/
	links.clear();
}

bool ApplicationBuilder::prepare_threads()
{
	if (threads_prepared) return true;
	pair<string,shared_ptr<NodeRecord> > node;
	BOOST_FOREACH(node,nodes) {
		log[debug] << "Preparing node " << node.first << endl;
		assert(node.second.get());
		assert(RegisteredClass::is_registered(node.second->type));
		shared_ptr<Instance> instance = RegisteredClass::prepare_instance(node.second->type);
		assert(instance.get());
		instance->params->merge(node.second->params);
		shared_ptr<Parameters> vars = assign_variables(node.second->variables);
		instance->params->merge(*vars);
		log[debug] << "Node " << node.first << " of class " << node.second->type
				<< " will be started with parameters:" << endl;
		show_params(*instance->params);
		try {
			threads[node.first] = instance->create_class(log,get_this_ptr());
		}
		catch (InitializationFailed &e) {
			log[error] << "Failed to create node '" << node.first << "' (" << node.second->type<<"): " << e.what() << endl;
			return false;
		}
		catch (Exception &e) {
			throw Exception(string("[")+node.first+string("] ")+string(e.what()));
		}
	}
	threads_prepared = true;
	return true;
}

void ApplicationBuilder::show_params(Parameters& _params,string prefix)
{
	pair<string,shared_ptr<Parameter> > par;
	BOOST_FOREACH(par,(_params.params)) {
		if (par.second->type==GroupType) {
			for (int i=0;;++i) {
				try {
					Parameters &p = (*par.second)[i];
					log[info] << prefix << par.first << "["<<i<<"] is group with following values:" << endl;
					show_params(p,prefix+'\t');
				}
				catch (OutOfRange &e) {
					break;
				}
			}
		} else {
			log[debug] << prefix << par.first << " has value " << par.second->get<string>() << endl;
		}
	}
}

bool ApplicationBuilder::prepare_links()
{
	pair<string,shared_ptr<LinkRecord> > link;
	BOOST_FOREACH(link,links) {
		log[debug] << "Preparing link " << link.first << endl;
		assert(link.second.get());
		shared_ptr<Parameters> params (new Parameters(*default_pipe_param));
		params->merge(link.second->params);
		shared_ptr<Parameters> vars = assign_variables(link.second->variables);
		params->merge(*vars);
		shared_ptr<BasicPipe> p = BasicPipe::generator(log,link.first,*params);
		threads[link.second->source_node]->connect_out(link.second->source_index,p);
		threads[link.second->target_node]->connect_in(link.second->target_index,p);
		pipes[link.first]=p;

	}
	return true;
}
bool ApplicationBuilder::spawn_threads()
{
	pair<string,shared_ptr<BasicIOThread> > thread;
	BOOST_FOREACH(thread,threads) {
		spawn_thread(thread.second);
		//log[info] << "Thread for object " << thread.first << " spawned as " << lexical_cast<string>(children[thread.second]->thread_ptr->native_handle()) << ", boost id: "<< children[thread.second]->thread_ptr->get_id()<< endl;
		tids[thread.first]=0;
	}
	return true;
}

bool ApplicationBuilder::stop_threads()
{
	return true;
}
bool ApplicationBuilder::delete_threads()
{
	return true;
}
bool ApplicationBuilder::delete_pipes()
{
	return true;
}

bool ApplicationBuilder::step()
{
	if (!start_time.is_not_a_date_time()) {
		if (start_time + run_limit < microsec_clock::local_time()) {
			log[info] << "Time limit reached (" << to_simple_string(run_limit)
					<< "), returning. Started " << to_simple_string(start_time.time_of_day())
					<< ", actual time is " << to_simple_string(microsec_clock::local_time().time_of_day())
					<< endl;
			return false;
		}
	}
	fetch_tids();
	return true;
}
shared_ptr<BasicIOThread> ApplicationBuilder::get_node(string id)
{
	if (threads.find(id)==threads.end()) {
		pair<string,shared_ptr<BasicIOThread> > p;
		BOOST_FOREACH(p,threads) {
			log[debug] << "I have " << p.first << endl;
		}
		throw Exception (string(id)+" does not exist");
	}
	return threads[id];
}

void ApplicationBuilder::init()
{
	string filename=params["config"].get<string>();
	if (filename!="") {
		if (!load_file(filename)) throw InitializationFailed(string("Failed to load file ")+filename);
	}
	module_dirs.push_back("./modules/");
	module_dirs.push_back("./bin/modules/");
	module_dirs.push_back("/usr/lib/yuri2/");
}

void ApplicationBuilder::init_local_params()
{
	int flags = 0;
	if 		(params["debug"].get<int>() >   1) flags|=verbose_debug;
	else if (params["debug"].get<int>() ==  1) flags|=debug;
	else if (params["debug"].get<int>() ==  0) flags|=normal;
	else if (params["debug"].get<int>() == -1) flags|=warning;
	else if (params["debug"].get<int>() <  -1) flags|=error;
	if 		(params["show_time"].get<bool>()) flags|=show_time;
	if 		(params["show_level"].get<bool>()) flags|=show_level;
	if 		(params["use_colors"].get<bool>()) flags|=use_colors;
	log.setFlags(flags);
}

void ApplicationBuilder::fetch_tids()
{
#ifdef __linux__
	pair<std::string,pid_t> tid_record;
	bool changed = false;
	BOOST_FOREACH(tid_record,tids) {
		if (!tid_record.second) {
			tids[tid_record.first] = threads[tid_record.first]->get_tid();
			log[debug] << "Thread " << tid_record.first << " has tid " << tids[tid_record.first] << endl;
			changed = true;
		}
	}
	if (changed) {
		std::string destdir = string("/tmp/")+lexical_cast<std::string>(getpid());
		boost::filesystem::path p(destdir);
		if (!boost::filesystem::exists(p)) {
			boost::filesystem::create_directory(p);
		}
		BOOST_FOREACH(tid_record,tids) {
			if (tid_record.second) {
				boost::filesystem::path p2 = p / lexical_cast<std::string>(tid_record.second); //(destdir+lexical_cast<std::string>(tid_record.second));
				if (!boost::filesystem::exists(p2)) {
					std::ofstream of(p2.string().c_str());
					of << tid_record.first << std::endl;
				}
			}
		}
	}
#endif
}

shared_ptr<Parameters> ApplicationBuilder::assign_variables(map<string,string> vars)
{
	pair<string,string> var;
	shared_ptr<Parameters> var_params(new Parameters());
	BOOST_FOREACH(var,vars) {
		if (variables.find(var.second) != variables.end()) {
			(*var_params)[var.first] = variables[var.second]->value;
		}
	}
	return var_params;
}

void ApplicationBuilder::parse_argv(vector<string> argv)
{
	log[debug] << "" << argv.size() << " arguments to parse" << endl;
	string arg;
	BOOST_FOREACH(arg,argv) {
		size_t delim = arg.find('=');
		if (delim==string::npos) {
			log[warning] << "Failed to parse argument '" <<arg<<"'" << endl;
		} else {
			string name, value;
			name = arg.substr(0,delim);
			value = arg.substr(delim+1);
			log[debug] << "Parsed argument " << name << " -> " << value << endl;
			if (variables.find(name) != variables.end()) {
				variables[name]->value=value;
			} else {
				variables[name]=shared_ptr<VariableRecord>(new VariableRecord(name,value));
			}
		}

	}
}

}

}
