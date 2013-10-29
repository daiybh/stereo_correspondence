/*
 * compressed_frame_types.h
 *
 *  Created on: 4.10.2013
 *      Author: neneko
 */

#ifndef COMPRESSED_FRAME_TYPES_H_
#define COMPRESSED_FRAME_TYPES_H_
#include "yuri/core/utils/new_types.h"
namespace yuri {
namespace core {
namespace compressed_frame {

const format_t unknown		= 0;

const format_t jpeg			= 0x10001;
const format_t mjpg			= 0x10002;
const format_t png			= 0x10003;
const format_t h264			= 0x10004;
const format_t dxt1			= 0x10005;
const format_t dxt5			= 0x10006;
const format_t vp8			= 0x10007;
const format_t dv			= 0x10008;
const format_t mpeg2		= 0x10009;

const format_t user_start 	= 0x11000;

}
}
}



#endif /* COMPRESSED_FRAME_TYPES_H_ */
