/*
 * register.cpp
 *
 *  Created on: 31.10.2013
 *      Author: neneko
 */

#include "GPUJpegDecoder.h"
#include "GPUJpegEncoder.h"
#include "yuri/core/thread/IOThreadGenerator.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/thread/ConverterRegister.h"

namespace yuri {
namespace gpujpeg {
using namespace yuri::core;
MODULE_REGISTRATION_BEGIN("gpujpeg")
		REGISTER_IOTHREAD("gpujpeg_decoder",GPUJpegDecoder)
		REGISTER_IOTHREAD("gpujpeg_encoder",GPUJpegEncoder)

		// High values because of it's incompatibility with mjpeg from logitech webcam...
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::rgb24, "gpujpeg_decoder", 125)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::yuv444, "gpujpeg_decoder", 120)
		REGISTER_CONVERTER(compressed_frame::jpeg, raw_format::uyvy422, "gpujpeg_decoder", 120)


		REGISTER_CONVERTER(raw_format::rgb24, compressed_frame::jpeg, "gpujpeg_encoder", 150)
		REGISTER_CONVERTER(raw_format::yuv444, compressed_frame::jpeg, "gpujpeg_encoder", 150)
		REGISTER_CONVERTER(raw_format::uyvy422, compressed_frame::jpeg, "gpujpeg_encoder", 150)

MODULE_REGISTRATION_END()


}
}
