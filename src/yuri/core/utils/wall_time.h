/*!
 * @file 		wall_time.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		2. 4. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_YURI_CORE_UTILS_WALL_CLOCK_H_
#define SRC_YURI_CORE_UTILS_WALL_CLOCK_H_
#include "yuri/core/utils/platform.h"
#include <ctime>
namespace yuri {
namespace core {
namespace utils {

std::tm get_current_local_time();
std::tm get_startup_local_time();


}
}
}



#endif /* SRC_YURI_CORE_UTILS_WALL_CLOCK_H_ */
