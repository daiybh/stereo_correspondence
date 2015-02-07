/*!
 * @file 		register.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		07.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */


#include "Saturate.h"
#include "Contrast.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace colors {

MODULE_REGISTRATION_BEGIN("colors")
		REGISTER_IOTHREAD("saturate", Saturate)
		REGISTER_IOTHREAD("contrast", Contrast)
MODULE_REGISTRATION_END()

}
}

