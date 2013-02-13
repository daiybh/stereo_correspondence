/*
 * test_module.cpp
 *
 *  Created on: 12.2.2013
 *      Author: neneko
 */

#include <iostream>
#include <dlfcn.h>
#include <boost/cstdint.hpp>

int main(int argc, char** argv)
{
//	using boost::uintptr_t;
	if (argc < 2) {
		std::cerr << "Usage " << argv[0] <<" <path-to-module>\n";
		return 1;
	}
	void *handle=dlopen(argv[1],RTLD_NOW|RTLD_GLOBAL);
	if (!handle) {
		std::cerr << "Failed to open file " << argv[1] <<": "<< dlerror() <<"\n";
		return 1;
	}
	// The ugly cast to uintptr_t is here for the sole purpose of silencing g++ warnings.
	const char * (*get_name)(void) = reinterpret_cast<const char * (*)(void)>(reinterpret_cast<uintptr_t>(dlsym(handle,"yuri_module_get_name")));
	void (*register_module)(void) = reinterpret_cast<void (*)(void)>(reinterpret_cast<uintptr_t>(dlsym(handle,"yuri_module_register")));
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
	dlclose(handle);
	if (!valid) {
		return 1;
	}
	return 0;
}


