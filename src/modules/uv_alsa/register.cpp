/*
 * register.cpp
 *
 *  Created on: 21.10.2013
 *      Author: neneko
 */
#include "yuri/core/Module.h"
#include "UVAlsaInput.h"
#include "UVAlsaOutput.h"

namespace yuri {


MODULE_REGISTRATION_BEGIN("uv_alsa")
		REGISTER_IOTHREAD("uv_alsa_input",uv_alsa_input::UVAlsaInput)
		REGISTER_IOTHREAD("uv_alsa_output",uv_alsa_output::UVAlsaOutput)
MODULE_REGISTRATION_END()


}


