/*!
 * @file 		register.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		14.12.2014
 * @copyright	Institute of Intermedia, 2014
 * 				Distributed BSD License
 *
 */


#include "Select.h"
#include "Unselect.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace select {


MODULE_REGISTRATION_BEGIN("select")
		REGISTER_IOTHREAD("select",Select)
		REGISTER_IOTHREAD("unselect",Unselect)
MODULE_REGISTRATION_END()


}
}
