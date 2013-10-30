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

convert::path_list find_conversion(format_t, format_t);



}
}

#endif /* CONVERTUTILS_H_ */
