/*!
 * @file 		RegisteredClass.h
 * @author 		Zdenek Travnicek
 * @date 		24.7.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef REGISTEREDCLASS_H_
#define REGISTEREDCLASS_H_

#include <map>
#include <string>
#include <vector>
#include "yuri/core/config_common.h"
#include "yuri/core/forward.h"
#include "yuri/exception/InitializationFailed.h"
#include "yuri/exception/NotImplemented.h"
#include "yuri/core/Instance.h"

#ifdef YURI_MODULE_IN_TREE
#define REGISTER(name,classname) namespace { yuri::core::RegisteredClassSpecialized<classname>  __reg__ (name); }
#else
#define REGISTER(name,classname) \
extern "C" { \
IMPORT const char * yuri_module_get_name() {return name;}\
IMPORT void 		yuri_module_register() {yuri::core::RegisteredClassSpecialized<classname>  *__reg__ = new yuri::core::RegisteredClassSpecialized<classname>(name);(void)__reg__;}\
}
#endif
namespace yuri {
namespace core {


typedef yuri::shared_ptr<class RegisteredClass> pRegisteredClass;

class EXPORT RegisteredClass {
public:
	typedef std::map<std::string,pRegisteredClass > register_map_t;
	RegisteredClass(std::string id) throw (Exception);
	virtual ~RegisteredClass();

	static bool is_registered(std::string id);
	static shared_ptr<std::vector<std::string> > list_registered();
	static pParameters get_params(std::string id);
	static pInstance prepare_instance(std::string id);
	static pInstance get_converter(long format_in, long format_out);

	static void reload_converters();
	static std::map<std::pair<long,long>, std::vector<shared_ptr<Converter > > > get_all_converters();
	static std::vector<shared_ptr<Converter > > get_converters(long in, long out);
	static std::vector<shared_ptr<Converter > > get_converters(std::pair<long, long> fmts);

	virtual pInstance _prepare_instance() = 0;
	virtual pParameters _get_params() = 0;
	virtual pInstance _prepare_converter(pInstance inst,
			long format_in, long format_out) = 0;
	static bool load_module(std::string path);
protected:
	static register_map_t *registeredClasses;
	static void add_to_register(std::string id, RegisteredClass*r);
	static bool do_is_registered(std::string id);
	static shared_ptr<std::vector<std::string> > do_list_registered();
	static pParameters do_get_params(std::string id);
	static pInstance do_prepare_instance(std::string id);
	static void do_add_to_register(std::string id, RegisteredClass*r);
	static std::map<std::pair<long,long>, std::vector<shared_ptr<Converter> > > converters;
	static mutex converters_mutex;
	static mutex *reg_mutex;
	static void do_reload_converters();
	static bool conv_sort(shared_ptr<Converter> a, shared_ptr<Converter> b);
	std::string id;
	pParameters params;
};

template<class T> class RegisteredClassSpecialized:public RegisteredClass {
public:
	RegisteredClassSpecialized(std::string id):
		RegisteredClass(id) {}

	typedef T type_id;
protected:
	virtual pInstance _prepare_instance();
	virtual pParameters _get_params();
	virtual pInstance _prepare_converter(pInstance inst,
			long format_in, long format_out);
};

template<class T> pParameters RegisteredClassSpecialized<T>::_get_params()
{
	if (params) return params;
	return params = type_id::configure();
}

template<class T> pInstance RegisteredClassSpecialized<T>::_prepare_instance()
{
	if (!_get_params()) throw exception::InitializationFailed("Failed to initialize parameters");
	pInstance inst (new Instance(id,type_id::generate,params));
	return inst;
}

template<class T> pInstance RegisteredClassSpecialized<T>::_prepare_converter(pInstance inst,
		long format_in, long format_out)
{
	if (!type_id::configure_converter(*(inst->params),format_in,format_out))
		throw exception::NotImplemented();
	return inst;
}

}

}

#endif /* REGISTEREDCLASS_H_ */

