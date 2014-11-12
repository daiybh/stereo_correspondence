/*!
 * @file 		builder_utils.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		9.11.2014
 * @copyright	CESNET, z.s.p.o, 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "builder_utils.h"
#include "yuri/core/utils/ModuleLoader.h"
#include "yuri/exception/InitializationFailed.h"

#include "yuri/core/thread/IOThreadGenerator.h"
#include "yuri/core/pipe/PipeGenerator.h"

namespace yuri {
namespace core{
namespace builder {


size_t load_builtin_modules(log::Log& l_)
{
	size_t loaded = 0;
	for (const auto& path: module_loader::get_builtin_paths()) {
		loaded += load_module_dir(l_, path);
	}
	return loaded;
}

size_t load_module_dir(log::Log& l_, const std::string& path)
{
	return load_modules(l_, module_loader::find_modules_path(path));
}

size_t load_modules(log::Log& l_, const std::vector<std::string>& modules)
{
	size_t loaded = 0;
	for (const auto& module: modules) {
		if (module_loader::load_module(module)) {
			l_[log::info] << "Loaded module " << module;
			++loaded;
		} else {
			l_[log::warning] << "Failed to load module " << module;
		}
	}
	return loaded;
}


void verify_node_class(const std::string& class_name)
{
	if (!verify_node_class(class_name, std::nothrow)) {
		throw exception::InitializationFailed("Node class " + class_name + " is not registered");
	}
}
bool verify_node_class(const std::string& class_name, const std::nothrow_t&)
{
	return IOThreadGenerator::get_instance().is_registered(class_name);}

void verify_link_class(const std::string& class_name)
{
	if (!verify_link_class(class_name, std::nothrow)) {
		throw exception::InitializationFailed("Node class " + class_name + " is not registered");
	}
}

bool verify_link_class(const std::string& class_name, const std::nothrow_t&)
{
	return PipeGenerator::get_instance().is_registered(class_name);
}


}
}
}

