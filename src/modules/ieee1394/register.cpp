/*!
 * @file 		register.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		24.2.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "DVSource.h"
#include "HDVSource.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace ieee1394 {
MODULE_REGISTRATION_BEGIN("ieee1394")
		REGISTER_IOTHREAD("dvsource",DVSource)
		REGISTER_IOTHREAD("hdvsource",HDVSource)
MODULE_REGISTRATION_END()
}
}



