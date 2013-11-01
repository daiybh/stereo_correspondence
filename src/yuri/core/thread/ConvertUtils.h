/*
 * ConvertUtils.h
 *
 *  Created on: 30.10.2013
 *      Author: neneko
 */

#ifndef CONVERTUTILS_H_
#define CONVERTUTILS_H_
#include "ConverterRegister.h"
#include <vector>
namespace yuri {
namespace core {

namespace convert {
struct convert_node_t {
	std::string name;
	format_t	source_format;
	format_t	target_format;
};

typedef std::vector<convert::convert_node_t> path_list;
}

/*!
 * Finds shortest way from @em source format to @em target format
 * @param source Input format
 * @param target Output format
 * @return pair containing the shortest path and it's cost
 */
std::pair<convert::path_list, size_t> find_conversion(format_t source, format_t target);



}
}

#endif /* CONVERTUTILS_H_ */
