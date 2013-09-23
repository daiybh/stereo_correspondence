/*
 * ModuleLoader.cpp
 *
 *  Created on: 15.9.2013
 *      Author: neneko
 */

#include "ModuleLoader.h"
#include "yuri/core/utils/platform.h"
#include "yuri/core/utils/make_unique.h"
#include "yuri/core/utils/DirectoryBrowser.h"
#include <stdexcept>
#include <iostream>

#if defined YURI_LINUX
#include <dlfcn.h>
#elif defined YURI_WINDOWS
#include <windows.h>
#else
#error Runtime object loading not supported on this platform
#endif

namespace yuri {
namespace core {
namespace module_loader {

const std::string module_prefix = "yuri2.8_module_";
const std::string module_get_name = "yuri2_8_module_get_name";
const std::string module_register = "yuri2_8_module_register";

struct dynamic_loader {
	dynamic_loader(const std::string& path);
	~dynamic_loader();
	void reset() { handle = 0;}
	template<typename T>
	T load_symbol(const std::string& symbol);
#if defined YURI_LINUX
	void* handle;
#elif defined YURI_WINDOWS
	HINSTANCE handle;
#endif
};

#if defined YURI_LINUX
dynamic_loader::dynamic_loader(const std::string& path)
{
	handle = dlopen(path.c_str(),RTLD_LAZY);
	if (!handle) throw std::runtime_error("Failed to open handle");
}
dynamic_loader::~dynamic_loader()
{
	if (handle) dlclose(handle);
}
template<typename T>
T dynamic_loader::load_symbol(const std::string& symbol)
{
	return reinterpret_cast<T>(reinterpret_cast<uintptr_t>(dlsym(handle,symbol.c_str())));
}
#elif defined YURI_WINDOWS
dynamic_loader::dynamic_loader(const std::string& path)
{
	handle = LoadLibrary(path.c_str());
	if (!handle) throw std::runtime_error("Failed to open handle");
}
dynamic_loader::~dynamic_loader()
{
	if (handle) FreeLibrary(handle);
}
template<typename T>
T dynamic_loader::load_symbol(const std::string& symbol)
{
	return reinterpret_cast<T>(reinterpret_cast<uintptr_t>(GetProcAddress(handle,symbol.c_str())));
}

#endif
namespace {
std::vector<std::string> built_in_paths = {
		"./modules",
		"./bin/modules",
#ifdef INSTALL_PREFIX
		INSTALL_PREFIX "/lib/yuri2/",
#else
		"/usr/lib/yuri2",
#endif
};
}
const std::vector<std::string>& get_builtin_paths()
{
	return built_in_paths;
}

std::vector<std::string> find_modules_path(const std::string& path)
{
	std::vector<std::string> modules;
	const auto& files = filesystem::browse_files(path, module_prefix);
	for (const auto& pp: files) {
//		std::cout << "x: " << pp << "\n";
		modules.push_back(pp);
	}
	return modules;
}

using get_name_t 		= const char * (*)(void);
using register_module_t = int (*)(void);

bool load_module(const std::string& path)
{
	try {
		get_name_t get_name = nullptr;
		register_module_t register_module = nullptr;

		dynamic_loader loader (path);

		// The ugly cast to uintptr_t is here for the sole purpose of silencing g++ warnings.
		get_name = loader.load_symbol<get_name_t>(module_get_name);
		register_module = loader.load_symbol<register_module_t>(module_register);

		if (get_name && register_module && register_module() == 0) {
			loader.reset(); // Effectively, we're leaking the handle here. Otherwise, it would be nearly impossible to prevent segfaults from released handles...
			return true;
		}
	}
	catch (std::runtime_error&) {}
	return false;
}


}
}
}




