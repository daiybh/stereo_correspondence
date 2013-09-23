/*
 * ModuleLoader.h
 *
 *  Created on: 15.9.2013
 *      Author: neneko
 */

#ifndef MODULELOADER_H_
#define MODULELOADER_H_
#include <string>
#include <vector>
#include "yuri/core/utils/new_types.h"
namespace yuri {
namespace core {
namespace module_loader {

struct module_handle;

std::vector<std::string> find_modules_path(const std::string& path);
bool load_module(const std::string& path);
const std::vector<std::string>& get_builtin_paths();

}
}
}



#endif /* MODULELOADER_H_ */
