/*!
 * @file 		register.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		5. 6. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */


#include "FlyCap.h"
#include "yuri/core/Module.h"
#include "yuri/core/thread/InputRegister.h"
namespace yuri {
namespace flycap {


MODULE_REGISTRATION_BEGIN("flycap")
		REGISTER_IOTHREAD("flycap",FlyCap)
		REGISTER_INPUT_THREAD("flycap", FlyCap::enumerate)
MODULE_REGISTRATION_END()

}
}

