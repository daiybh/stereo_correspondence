/*
 * register.cpp
 *
 *  Created on: 31.10.2013
 *      Author: neneko
 */

#include "JpegDecoder.h"
#include "JpegEncoder.h"
#include "yuri/core/thread/IOThreadGenerator.h"
namespace yuri {
namespace jpeg {

MODULE_REGISTRATION_BEGIN("jpeg")
		REGISTER_IOTHREAD("jpeg_decoder",JpegDecoder)
		REGISTER_IOTHREAD("jpeg_encoder",JpegEncoder)
MODULE_REGISTRATION_END()


}
}
