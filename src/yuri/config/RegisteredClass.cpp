/*
 * RegisteredClass.cpp
 *
 *  Created on: Jul 24, 2010
 *      Author: worker
 */

#include "RegisteredClass.h"
#ifdef __linux__
#include <dlfcn.h>
#else
#warning "Runtime object loading not supported on this platform"
#endif
namespace yuri {

namespace config {

RegisteredClass::register_map_t *RegisteredClass::registeredClasses = 0;
std::map<std::pair<long,long>, std::vector<shared_ptr<Converter> > > RegisteredClass::converters;
mutex RegisteredClass::converters_mutex;
mutex *RegisteredClass::reg_mutex = 0;

#include <iostream>
RegisteredClass::RegisteredClass(std::string id)
			throw (Exception):
		id(id)

{
	//std::cout << "REG " << id << endl;
	add_to_register(id, this);
}

RegisteredClass::~RegisteredClass() {
	//registeredClasses->erase(this->id);
}


bool RegisteredClass::is_registered(std::string id)
{
	if (!reg_mutex) reg_mutex=new boost::mutex();
	mutex::scoped_lock l(*reg_mutex);
	return do_is_registered(id);
}

void RegisteredClass::add_to_register(std::string id, RegisteredClass *r)
{
	if (!reg_mutex) reg_mutex=new boost::mutex();
	mutex::scoped_lock l(*reg_mutex);
	do_add_to_register(id,r);
}


shared_ptr<std::vector<std::string> > RegisteredClass::list_registered()
{
	if (!reg_mutex) reg_mutex=new boost::mutex();
	mutex::scoped_lock l(*reg_mutex);
	return do_list_registered();
}

shared_ptr<Parameters> RegisteredClass::get_params(std::string id)
{
	if (!reg_mutex) reg_mutex=new boost::mutex();
	mutex::scoped_lock l(*reg_mutex);
	return do_get_params(id);
}


shared_ptr<Instance> RegisteredClass::prepare_instance(std::string id)
{
	if (!reg_mutex) reg_mutex=new boost::mutex();
	mutex::scoped_lock l(*reg_mutex);
	return do_prepare_instance(id);
}

std::map<std::pair<long,long>, std::vector<shared_ptr<Converter > > > RegisteredClass::get_all_converters()
{
	mutex::scoped_lock l(converters_mutex);
	if (converters.empty()) do_reload_converters();
	return converters;
}
std::vector<shared_ptr<Converter > > RegisteredClass::get_converters(long in, long out)
{
	return get_converters(std::make_pair(in,out));
}
std::vector<shared_ptr<Converter > > RegisteredClass::get_converters(std::pair<long, long> fmts)
{
	mutex::scoped_lock l(converters_mutex);
	std::vector<shared_ptr<Converter> > v;
	if (converters.empty()) do_reload_converters();
	if (converters.find(fmts) == converters.end()) return v;
	return converters[fmts];
}

void RegisteredClass::reload_converters()
{
	mutex::scoped_lock l(converters_mutex);
	do_reload_converters();
}
void RegisteredClass::do_reload_converters()
{
	std::pair<std::string,shared_ptr<RegisteredClass> > p;
	BOOST_FOREACH(p,*registeredClasses) {
		shared_ptr<Parameters> params = p.second->_get_params();
		std::pair<std::pair<long,long>, shared_ptr<Converter> > conv;
		if (params) {
			BOOST_FOREACH(conv,params->get_converters()) {
				conv.second->id = p.first;
				converters[conv.first].push_back(conv.second);
			}
		}
	}
}

bool RegisteredClass::conv_sort(shared_ptr<Converter> a, shared_ptr<Converter> b)
{
	return (a->confidence > b->confidence);
}
shared_ptr<Instance> RegisteredClass::get_converter(long format_in,
		long format_out)
{
	std::vector<shared_ptr<Converter> > converters;
	converters = get_converters(format_in, format_out);
	if (!converters.empty()) { // Direct conversion available
		sort(converters.begin(),converters.end(),RegisteredClass::conv_sort);
		shared_ptr<Converter> conv;
		BOOST_FOREACH(conv, converters) {
			if (!is_registered(conv->id)) {
				continue;
			} else {
				try {
					shared_ptr<Instance> inst = prepare_instance(conv->id);
					mutex::scoped_lock l(converters_mutex);
					return (*registeredClasses)[conv->id]->_prepare_converter(
							inst, format_in, format_out);
				}
				catch (Exception &e) {
					continue;
				}
			}

		}
	}
	// We didn't find a simple conversion.
	// It is either not defined or does not work
/*	// Let's try to find more complicated conversion.
	std::set<long> sources,new_sources,new_sources2;
	std::map<long,long> paths;
	std::pair<long,long> tmp_pair;
	long last_step = YURI_FMT_NONE;
	mutex::scoped_lock l(converters_mutex);
	sources.insert(format_in);
	new_sources.insert(format_in);
	paths[format_in]=YURI_FMT_NONE;
	while (true) {
		cout << "Pass begin" << endl;
		BOOST_FOREACH(tmp_pair,converters) {
			if (new_sources.find(tmp_pair.first) != new_sources.end()) {
				if (tmp_pair.second == format_out) {
					last_step = tmp_pair.first;
					break;
				}
				if (sources.find(tmp_pair.second)==sources.end()) {

				}
			}
		}
	}
*/

	throw NotImplemented("No usable converter found!");
}


bool RegisteredClass::do_is_registered(std::string id)
{
	if (registeredClasses->find(id) ==  registeredClasses->end()) return false;
	return true;
}
shared_ptr<std::vector<std::string> > RegisteredClass::do_list_registered()
{
	std::pair<std::string,shared_ptr<RegisteredClass> > p;
	shared_ptr<std::vector<std::string> > regs (new std::vector<std::string>);
	//std::cout << "class object exists: " << (bool)(registeredClasses) << std::endl;
	if (registeredClasses) {
		//std::cout << "Number of classes: " << registeredClasses->size() << endl;
		BOOST_FOREACH(p,*registeredClasses) {
			regs->push_back(p.first);
		}
	}
	//regs->push_back(registeredClasses->front());
	//return regs;
	return regs;
}
shared_ptr<Parameters> RegisteredClass::do_get_params(std::string id)
{
	if (do_is_registered(id)) {
		return (*registeredClasses)[id]->_get_params();
	}
	throw Exception("Requested params for non-existing class!!");
}

shared_ptr<Instance> RegisteredClass::do_prepare_instance(std::string id)
{
	if (do_is_registered(id)) {
		return (*registeredClasses)[id]->_prepare_instance();
	}
	throw Exception ("Requested instance for non-registered class!");
}

void RegisteredClass::do_add_to_register(std::string id, RegisteredClass*r)
{
//	std::cout <<  "Registering " << id << endl;
	if (!registeredClasses) {
		registeredClasses = new register_map_t();
	}
	if (registeredClasses->find(id) !=  registeredClasses->end())
		throw Exception("Tried to register already registered class!");
	shared_ptr<RegisteredClass> rc(r);
	(*registeredClasses)[id] = rc;
}


bool RegisteredClass::load_module(std::string path)
{
#ifdef __linux__
	void *handle=dlopen(path.c_str(),RTLD_LAZY);
	if (!handle) return false;
	// The ugly cast to uintptr_t is here for the sole purpose of silencing g++ warnings.
	const char * (*get_name)(void) = reinterpret_cast<const char * (*)(void)>(reinterpret_cast<uintptr_t>(dlsym(handle,"yuri_module_get_name")));
	void (*register_module)(void) = reinterpret_cast<void (*)(void)>(reinterpret_cast<uintptr_t>(dlsym(handle,"yuri_module_register")));

	bool valid = true;
	if (!get_name || !register_module) {
		valid = false;
		std::cerr << "Module doesn't export libyuri2 interface\n";
	}
	if (valid && RegisteredClass::is_registered(get_name())) {
		std::cerr << "Module already registered\n";
		valid = false;
	}
	if (!valid) {//	std::cout << "Not a valid module\n";
		dlclose(handle);
		return false;
	}
	register_module();

	return true;
#else
	return false;
#endif
}
}


}
