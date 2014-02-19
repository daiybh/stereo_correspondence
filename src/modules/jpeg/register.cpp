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
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::bgr24, "jpeg_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::yuv444, "jpeg_decoder", 25)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::y8, "jpeg_decoder", 35)

		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::rgba32, "jpeg_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::argb32, "jpeg_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::bgra32, "jpeg_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::abgr32, "jpeg_decoder", 30)


		REGISTER_CONVERTER(raw_format::rgb24, compressed_frame::jpeg, "jpeg_encoder", 200)
		REGISTER_CONVERTER(raw_format::bgr24, compressed_frame::jpeg, "jpeg_encoder", 200)
		REGISTER_CONVERTER(raw_format::rgba32, compressed_frame::jpeg, "jpeg_encoder", 200)
		REGISTER_CONVERTER(raw_format::bgra32, compressed_frame::jpeg, "jpeg_encoder", 200)
		REGISTER_CONVERTER(raw_format::abgr32, compressed_frame::jpeg, "jpeg_encoder", 200)
		REGISTER_CONVERTER(raw_format::argb32, compressed_frame::jpeg, "jpeg_encoder", 200)
		REGISTER_CONVERTER(raw_format::yuv444, compressed_frame::jpeg, "jpeg_encoder", 200)
		REGISTER_CONVERTER(raw_format::y8, compressed_frame::jpeg, "jpeg_encoder", 200)

MODULE_REGISTRATION_END()


}
}
