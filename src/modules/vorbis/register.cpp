/*
 * register.cpp
 *
 *  Created on: 25.3.2014
 *      Author: neneko
 */

#include "VorbisEncoder.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace vorbis {

MODULE_REGISTRATION_BEGIN("vorbis")
		REGISTER_IOTHREAD("vorbis_encoder",VorbisEncoder)
MODULE_REGISTRATION_END()

}
}

