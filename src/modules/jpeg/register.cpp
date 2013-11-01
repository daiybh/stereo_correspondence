/*
 * register.cpp
 *
 *  Created on: 31.10.2013
 *      Author: neneko
 */

#include "JpegDecoder.h"
#include "JpegEncoder.h"
#include "yuri/core/thread/IOThreadGenerator.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/thread/ConverterRegister.h"

namespace yuri {
namespace jpeg {
using namespace yuri::core;
MODULE_REGISTRATION_BEGIN("jpeg")
		REGISTER_IOTHREAD("jpeg_decoder",JpegDecoder)
		REGISTER_IOTHREAD("jpeg_encoder",JpegEncoder)

		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::rgb24, "jpeg_decoder", 30)
//		REGISTER_CONVERTER(compressed_frame::mjpg, raw_format::rgb24, "jpeg_decoder", 30)
MODULE_REGISTRATION_END()


}
}
