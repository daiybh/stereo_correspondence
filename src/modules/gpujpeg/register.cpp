/*
 * register.cpp
 *
 *  Created on: 31.10.2013
 *      Author: neneko
 */

#include "GPUJpegDecoder.h"
//#include "GPUJpegEncoder.h"
#include "yuri/core/thread/IOThreadGenerator.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/thread/ConverterRegister.h"

namespace yuri {
namespace gpujpeg {
using namespace yuri::core;
MODULE_REGISTRATION_BEGIN("gpujpeg")
		REGISTER_IOTHREAD("gpujpeg_decoder",GPUJpegDecoder)
//		REGISTER_IOTHREAD("gpujpeg_encoder",JpegEncoder)

		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::rgb24, "jpeg_decoder", 25)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::yuv444, "jpeg_decoder", 20)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::yuyv422, "jpeg_decoder", 20)
		//REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::y8, "jpeg_decoder", 35)
#if defined(JCS_EXTENSIONS) && defined(JCS_ALPHA_EXTENSIONS)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::bgr24, "jpeg_decoder", 30)

		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::rgba32, "jpeg_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::argb32, "jpeg_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::bgra32, "jpeg_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::abgr32, "jpeg_decoder", 30)
#endif

//		REGISTER_CONVERTER(raw_format::rgb24, compressed_frame::jpeg, "jpeg_encoder", 200)
#if defined(JCS_EXTENSIONS) && defined(JCS_ALPHA_EXTENSIONS)
		REGISTER_CONVERTER(raw_format::bgr24, compressed_frame::jpeg, "jpeg_encoder", 200)
		REGISTER_CONVERTER(raw_format::rgba32, compressed_frame::jpeg, "jpeg_encoder", 200)
		REGISTER_CONVERTER(raw_format::bgra32, compressed_frame::jpeg, "jpeg_encoder", 200)
		REGISTER_CONVERTER(raw_format::abgr32, compressed_frame::jpeg, "jpeg_encoder", 200)
		REGISTER_CONVERTER(raw_format::argb32, compressed_frame::jpeg, "jpeg_encoder", 200)
#endif
//		REGISTER_CONVERTER(raw_format::yuv444, compressed_frame::jpeg, "jpeg_encoder", 200)
//		REGISTER_CONVERTER(raw_format::y8, compressed_frame::jpeg, "jpeg_encoder", 200)

MODULE_REGISTRATION_END()


}
}
