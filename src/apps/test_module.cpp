/*
 * test_module.cpp
 *
 *  Created on: 12.2.2013
 *      Author: neneko
 */

#include <iostream>
#include <boost/cstdint.hpp>
#if defined __linux__
#include <dlfcn.h>
#elif defined _WIN32
#include <windows.h>
#endif


int main(int argc, char** argv)
{
//	using boost::uintptr_t;
	if (argc < 2) {
		std::cerr << "Usage " << argv[0] <<" <path-to-module>\n";
		return 1;
	}
#if defined __linux__
	void *handle=dlopen(argv[1],RTLD_NOW|RTLD_GLOBAL);
#elif defined _WIN32
	HINSTANCE handle = LoadLibrary(argv[1]);
#endif
	if (!handle) {
		std::cerr << "Failed to open file " << argv[1] 
#if defined __linux__
			<<": " << dlerror() 
#endif
			<<"\n";
		return 1;
	}
	// The ugly cast to uintptr_t is here for the sole purpose of silencing g++ warnings.
	typedef const char * (*get_name_t)(void);
	typedef void (*register_module_t)(void);
#if defined __linux
	get_name_t get_name = reinterpret_cast<get_name_t>(reinterpret_cast<uintptr_t>(dlsym(handle,"yuri_module_get_name")));
	register_module_t register_module = reinterpret_cast<register_module_t>(reinterpret_cast<uintptr_t>(dlsym(handle,"yuri_module_register")));
#elif defined _WIN32
	get_name_t get_name = reinterpret_cast<get_name_t>(GetProcAddress(handle,"yuri_module_get_name"));
	register_module_t register_module = reinterpret_cast<register_module_t>(GetProcAddress(handle,"yuri_module_register"));
#endif
	bool valid = true;
	if (!get_name) {
		std::cerr << "Module doesn't export yuri_module_get_name\n";
		valid = false;
	}
	if (!register_module) {
		std::cerr << "Module doesn't export yuri_module_register\n";
		valid = false;
	}
	if (valid) {
		const char* name = get_name();
		if (!name) {
			std::cerr << "Module doesn't return it's name\n";
			valid = false;
		} else {
			std::cerr << "Module " << name << " seem valid\n";
		}
	}
#if defined __linux
	dlclose(handle);
#elif defined _WIN32
	FreeLibrary(handle);
#endif
	if (!valid) {
		return 1;
	}
	return 0;
}


