/*
 * register.cpp
 *
 *  Created on: 29.3.2014
 *      Author: neneko
 */

#include "X264Encoder.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace x264 {


MODULE_REGISTRATION_BEGIN("x264")
		REGISTER_IOTHREAD("x264_encoder",X264Encoder)
MODULE_REGISTRATION_END()

}

}
