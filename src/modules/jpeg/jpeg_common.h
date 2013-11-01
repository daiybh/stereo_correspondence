/*
 * jpeg_common.h
 *
 *  Created on: 1.11.2013
 *      Author: neneko
 */

#ifndef JPEG_COMMON_H_
#define JPEG_COMMON_H_
#include "yuri/core/utils/new_types.h"
#include <jpeglib.h>

namespace yuri {
namespace jpeg {

format_t jpeg_to_yuri(J_COLOR_SPACE colspace);
J_COLOR_SPACE  yuri_to_jpeg(format_t fmt);

}
}

#endif /* JPEG_COMMON_H_ */
