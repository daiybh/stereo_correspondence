/*!
 * @file 		ModuleLoader.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		15.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
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

EXPORT std::vector<std::string> find_modules_path(const std::string& path);
EXPORT bool load_module(const std::string& path);
EXPORT const std::vector<std::string>& get_builtin_paths();

}
}
}



#endif /* MODULELOADER_H_ */
