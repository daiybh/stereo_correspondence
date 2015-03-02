/*!
 * @file 		v4l2_common.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		01.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */


#ifndef V4L2_COMMON_H_
#define V4L2_COMMON_H_

#include <vector>
#include "yuri/core/utils/uvector.h"
#include <string>

namespace yuri {
namespace v4l2 {

/** Structure to hold buffer informations */
struct buffer_t {
		uvector<uint8_t> data;
};

/** Methods to read from v4l2 devices*/
enum class capture_method_t {
	/** No known way how to read from device*/
	none,
	/** Use mmap to map buffers */
	mmap,
	/** Use user specified buffers */
	user,
	/** Use direct read from the device file */
	read
};

enum class control_support_t {
	supported,
	not_supported,
	disabled,
};

struct control_state_t {
	control_support_t supported;
	int32_t value;
	int32_t min_value;
	int32_t max_value;
	std::string name;
};
}
}


#endif /* V4L2_COMMON_H_ */
