/*
 * register.cpp
 *
 *  Created on: 2.11.2013
 *      Author: neneko
 */

#include "PngEncoder.h"
#include "PngDecoder.h"
#include "yuri/core/thread/IOThreadGenerator.h"
#include "yuri/core/thread/ConverterRegister.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/compressed_frame_types.h"
namespace yuri {
namespace png {
using namespace yuri::core;
MODULE_REGISTRATION_BEGIN("png")
		REGISTER_IOTHREAD("png_decoder",PngDecoder)
		REGISTER_IOTHREAD("png_encoder",PngEncoder)

		REGISTER_CONVERTER(raw_format::y8, 		compressed_frame::png, "png_encoder", 200)
		REGISTER_CONVERTER(raw_format::y16, 	compressed_frame::png, "png_encoder", 200)
		REGISTER_CONVERTER(raw_format::rgb24, 	compressed_frame::png, "png_encoder", 200)
		REGISTER_CONVERTER(raw_format::rgb48, 	compressed_frame::png, "png_encoder", 200)
		REGISTER_CONVERTER(raw_format::bgr24, 	compressed_frame::png, "png_encoder", 200)
		REGISTER_CONVERTER(raw_format::bgr48, 	compressed_frame::png, "png_encoder", 200)

		REGISTER_CONVERTER(raw_format::rgba32, 	compressed_frame::png, "png_encoder", 200)
		REGISTER_CONVERTER(raw_format::rgba64, 	compressed_frame::png, "png_encoder", 200)
		REGISTER_CONVERTER(raw_format::bgra32, 	compressed_frame::png, "png_encoder", 200)
		REGISTER_CONVERTER(raw_format::bgra64,	compressed_frame::png, "png_encoder", 200)


		REGISTER_CONVERTER(compressed_frame::png, raw_format::y16,		"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::y8, 		"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::bgr48, 	"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::rgb48, 	"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::bgr24,	"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::rgb24,	"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::abgr64, 	"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::argb64, 	"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::abgr32, 	"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::argb32, 	"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::bgra64, 	"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::rgba64, 	"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::bgra32, 	"png_decoder", 30)
		REGISTER_CONVERTER(compressed_frame::png, raw_format::rgba32, 	"png_decoder", 30)

MODULE_REGISTRATION_END()


}
}

