/*
 * SpecialPipes.cpp
 *
 *  Created on: 9.9.2013
 *      Author: neneko
 */
#include "SpecialPipes.h"

namespace yuri {
namespace core {

//	REGISTER_PIPE("unlimited_blocking",		BlockingUnlimitedPipe)
	REGISTER_PIPE("unlimited",				NonBlockingUnlimitedPipe)
	REGISTER_PIPE("single_blocking",		BlockingSingleFramePipe)
	REGISTER_PIPE("single",					NonBlockingSingleFramePipe)
	REGISTER_PIPE("count_limited_blocking",	BlockingCountLimitedPipe)
	REGISTER_PIPE("count_limited",			NonBlockingCountLimitedPipe)
	REGISTER_PIPE("size_limited_blocking",	BlockingSizeLimitedPipe)
	REGISTER_PIPE("size_limited",			NonBlockingSizeLimitedPipe)
}
}


