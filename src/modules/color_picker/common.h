/*!
 * @file 		common.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		24. 11. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_MODULES_COLOR_PICKER_COMMON_H_
#define SRC_MODULES_COLOR_PICKER_COMMON_H_
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/utils/color.h"
namespace yuri {
namespace color_picker {

std::tuple<core::pRawVideoFrame, core::color_t>
process_rect(const core::pRawVideoFrame& frame, const geometry_t& geometry, bool show_color);

}
}



#endif /* SRC_MODULES_COLOR_PICKER_COMMON_H_ */
