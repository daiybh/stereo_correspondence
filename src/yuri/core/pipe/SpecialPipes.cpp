/*!
 * @file 		SpecialPipes.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		9.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "SpecialPipes.h"

namespace yuri {
namespace core {

//	REGISTER_PIPE("unlimited_blocking",         	BlockingUnlimitedPipe)
    REGISTER_PIPE("unlimited",                      NonBlockingUnlimitedPipe)
    REGISTER_PIPE("single_blocking",                BlockingSingleFramePipe)
    REGISTER_PIPE("single",                         NonBlockingSingleFramePipe)
    REGISTER_PIPE("count_limited_blocking",         BlockingCountLimitedPipe)
    REGISTER_PIPE("count_limited",                  NonBlockingCountLimitedPipe)
    REGISTER_PIPE("size_limited_blocking",          BlockingSizeLimitedPipe)
    REGISTER_PIPE("size_limited",                   NonBlockingSizeLimitedPipe)
    REGISTER_PIPE("unreliable_single_blocking",		BlockingUnreliableSingleFramePipe)
    REGISTER_PIPE("unreliable_single",              NonBlockingUnreliableSingleFramePipe)
}
}


