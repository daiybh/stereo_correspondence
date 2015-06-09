/*!
 * @file 		environment.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		07.06.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "environment.h"
#include "yuri/core/utils/string.h"

#include <stdlib.h>
namespace yuri {
namespace core {
namespace utils {

std::string get_environment_variable(const std::string& name, const std::string& def_value)
{
	if (auto v = ::getenv(name.c_str())) {
		return v;
	}
	return def_value;
}
std::vector<std::string> get_environment_path(const std::string& name)
{
#ifdef YURI_WIN
	const char delimiter = ';';
#else
	const char delimiter = ':';
#endif
	auto v = get_environment_variable(name);
	if (!v.empty()) {
		return split_string(v, delimiter);
	}
	return {};

}

}
}
}
