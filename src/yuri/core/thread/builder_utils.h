/*!
 * @file 		builder_utils.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		9.11.2014
 * @copyright	CESNET, z.s.p.o, 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef BUILDER_UTILS_H_
#define BUILDER_UTILS_H_
#include <string>
#include <vector>
#include <new>
#include "yuri/log/Log.h"

namespace yuri {
namespace core{
namespace builder {

size_t load_builtin_modules(log::Log& l_);
size_t load_module_dir(log::Log& l_, const std::string& path);
size_t load_modules(log::Log& l_, const std::vector<std::string>& modules);

void verify_node_class(const std::string& node);
bool verify_node_class(const std::string& node, const std::nothrow_t& tag);
void verify_link_class(const std::string& node);
bool verify_link_class(const std::string& node, const std::nothrow_t& tag);



}
}
}


#endif /* BUILDER_UTILS_H_ */
