/*!
 * @file 		frei0r_module.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		05.06.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FREI0R_MODULE_H_
#define FREI0R_MODULE_H_
#include "yuri/core/utils/ModuleLoader.h"
#include "yuri/exception/InitializationFailed.h"

#include <frei0r.h>

namespace yuri {
namespace frei0r {


using f0r_init_t = decltype(&::f0r_init);
using f0r_deinit_t = decltype(&::f0r_deinit);
using f0r_get_plugin_info_t = decltype(&::f0r_get_plugin_info);
using f0r_construct_t = decltype(&::f0r_construct);
using f0r_destruct_t = decltype(&::f0r_destruct);
using f0r_update_t = decltype(&::f0r_update);
using f0r_get_param_info_t = decltype(&::f0r_get_param_info);
using f0r_get_param_value_t = decltype(&::f0r_get_param_value);
using f0r_set_param_value_t = decltype(&::f0r_set_param_value);



struct frei0r_module_t {
	frei0r_module_t (const std::string& path):
	handle(path)
	{
		load_symbol(init, "f0r_init");
		if (!init ) {
			throw exception::InitializationFailed("Invalid frei0r module (missing init)");
		}
		load_symbol(get_plugin_info, "f0r_get_plugin_info");
		if (!get_plugin_info) {
			throw exception::InitializationFailed("Invalid frei0r module (missing get_plugin_info)");
		}
		init();
		load_symbol(construct, "f0r_construct");
		if (!construct) {
			throw exception::InitializationFailed("Invalid frei0r module (missing construct)");
		}
		load_symbol(destruct, "f0r_destruct");
		if (!destruct) {
			throw exception::InitializationFailed("Invalid frei0r module (missing destruct)");
		}
		get_plugin_info(&info);
		load_symbol(get_param_info, "f0r_get_param_info");
		load_symbol(deinit, "f0r_deinit");
		load_symbol(get_param_value, "f0r_get_param_value");
		load_symbol(set_param_value, "f0r_set_param_value");
		load_symbol(update, "f0r_update");
	}
	~frei0r_module_t() noexcept
	{
		if (deinit) deinit();
	}

	template<class T>
	void load_symbol(T& func, const std::string name)
	{
		func = handle.load_symbol<T>(name);
	}

	core::module_loader::dynamic_loader  handle;
	f0r_plugin_info_t info;


	f0r_init_t init;
	f0r_deinit_t deinit;
	f0r_get_plugin_info_t get_plugin_info;
	f0r_get_param_info_t get_param_info;
	f0r_construct_t construct;
	f0r_destruct_t destruct;
	f0r_get_param_value_t get_param_value;
	f0r_set_param_value_t set_param_value;
	f0r_update_t update;
};

}
}

#endif /* FREI0R_MODULE_H_ */
