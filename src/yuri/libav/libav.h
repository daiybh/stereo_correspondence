/*
 * libav.h
 *
 *  Created on: 29.10.2013
 *      Author: neneko
 */

#ifndef LIBAV_H_
#define LIBAV_H_
#include "yuri/core/utils/new_types.h"
extern "C" {
	#include "libavcodec/avcodec.h"
}
namespace yuri {
namespace libav {

PixelFormat avpixelformat_from_yuri(yuri::format_t format);
CodecID avcodec_from_yuri_format(yuri::format_t codec);

yuri::format_t yuri_pixelformat_from_av(PixelFormat format);
yuri::format_t yuri_format_from_avcodec(CodecID codec);
}
}


#endif /* LIBAV_H_ */
