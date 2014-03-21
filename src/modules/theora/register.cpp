/*!
 * @file 		theora.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		21.3.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */



#include "TheoraEncoder.h"
#include "yuri/core/Module.h"
namespace yuri {
namespace theora {

MODULE_REGISTRATION_BEGIN("theora")
		REGISTER_IOTHREAD("theora_encoder",TheoraEncoder)
MODULE_REGISTRATION_END()

}

}
