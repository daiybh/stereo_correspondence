/*!
 * @file 		global_time.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		19.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "global_time.h"
namespace yuri {
namespace core {
namespace utils {

namespace {
const timestamp_t global_time_start;
}

timestamp_t get_global_start_time()
{
	return global_time_start;
}


}
}
}


