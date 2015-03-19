/*!
 * @file 		frame_info.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		19. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_YURI_CORE_UTILS_FRAME_INFO_H_
#define SRC_YURI_CORE_UTILS_FRAME_INFO_H_

#include <string>
#include "new_types.h"

namespace yuri {
namespace core {
namespace utils {

std::string get_frame_type_name(format_t fmt, bool short_name);
}
}
}
#endif /* SRC_YURI_CORE_UTILS_FRAME_INFO_H_ */
